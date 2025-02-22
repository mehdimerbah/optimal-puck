"""
Replay Buffer mit prioritätsbasierter Pufferung (Dreamer-artig).

Funktionalitäten:
  - Speicherung von Transitionen in Chunks
  - Prioritätensampling (Prioritized oder Uniform)
  - Sampling variabler Sequenzlängen über chunk-Grenzen hinweg
  - Thread-sicher durch Locking
  - Verhindert, dass leere Chunks im Sampler landen
"""

import threading
import time
import collections
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import traceback
import uuid
from pathlib import Path
import elements


# ---------------------------------------------------------------------
# 1) Hilfsklassen (Dummy-Implementierungen) für Lock, Timer, UUID usw.
# ---------------------------------------------------------------------

class RWLock:
    """Sehr vereinfachtes R/W-Lock (in Wirklichkeit nur ein reentrantes Lock)."""
    def __init__(self):
        self._lock = threading.RLock()
    def reading(self):
        return self._lock
    def writing(self):
        return self._lock


class Chunk:
    """
    Speichert bis zu `size` Steps. Erst nach dem ersten .append() wird
    self.data allokiert. Bleibt length=0, so bleibt self.data=None.
    """
    __slots__ = ('time', 'uuid', 'succ', 'length', 'size', 'data', 'saved')

    def __init__(self, size=1024):
        self.time = int(time.time() * 1000)
        self.uuid = uuid.uuid4().bytes
        self.succ = uuid.UUID(int=0).bytes   # Nachfolgerchunk (anfangs 0)
        self.length = 0
        self.size = size
        self.data = None
        self.saved = False

    @property
    def filename(self):
        """Bildet den Dateinamen aus time-uuid-succ-length"""
        succ_hex = uuid.UUID(bytes=self.succ).hex
        uuid_hex = uuid.UUID(bytes=self.uuid).hex
        return f'{self.time}-{uuid_hex}-{succ_hex}-{self.length}.npz'

    @property
    def nbytes(self):
        """Geschätzte Größe der Daten im RAM."""
        if self.data is None:
            return 0
        return sum(arr.nbytes for arr in self.data.values())

    def append(self, step):
        """Hängt einen Step an das Ende dieses Chunks an."""
        if self.length >= self.size:
            raise ValueError("Chunk is already full.")
        if self.data is None:
            # Allokation anhand der Shapes des ersten Steps:
            self.data = {
                k: np.empty((self.size, *np.asarray(v).shape), np.asarray(v).dtype)
                for k, v in step.items()
            }
        # Kopiere die Daten ins Array:
        for k, v in step.items():
            self.data[k][self.length] = v
        self.length += 1

    def slice(self, index, length):
        """
        Schneidet [index:index+length] aus diesem Chunk heraus und gibt
        ein Dict {key: array} zurück. Wenn keine Daten vorhanden (self.data=None),
        oder index >= self.length, oder length=0, kann ein leeres Dict zurückkommen.
        """
        if self.data is None:
            # Noch nie ein Step appended -> kein Inhalt
            # Print-Debug:
            print(f"[DEBUG] slice() aufgerufen, aber self.data ist None! index={index}, length={length}")
            return {}
        if index >= self.length or length <= 0:
            # Nichts zu holen
            return {}
        out = {}
        stop = min(index + length, self.length)
        for k, arr in self.data.items():
            out[k] = arr[index:stop]
        return out

    def save(self, directory, log=False):
        """Speichert die vorhandenen Steps in eine komprimierte .npz-Datei."""
        if self.saved:
            return
        self.saved = True
        filepath = Path(directory) / self.filename
        data_to_save = {k: self.data[k][:self.length] for k in self.data}
        with open(filepath, 'wb') as f:
            np.savez_compressed(f, **data_to_save)
        if log:
            print(f"Saved chunk: {filepath.name}")

    @classmethod
    def load(cls, filename, error='raise'):
        """Lädt ein Chunk aus einer .npz-Datei."""
        try:
            with open(filename, 'rb') as f:
                npz = np.load(f)
                data = {k: npz[k] for k in npz.files}
        except Exception:
            tb = traceback.format_exc()
            print(f"Error loading chunk {filename}:\n{tb}")
            if error == 'raise':
                raise
            else:
                return None

        chunk = cls()
        parts = filename.stem.split('-')
        chunk.time = int(parts[0])
        chunk.uuid = uuid.UUID(parts[1]).bytes
        chunk.succ = uuid.UUID(parts[2]).bytes
        chunk.length = int(parts[3])
        chunk.data = data
        chunk.saved = True
        return chunk


class Uniform:
    """Einfacher Uniform-Sampler für Replay."""
    def __init__(self, seed=0):
        self.keys = []
        self.indices = {}
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.keys)

    def __call__(self):
        with self.lock:
            index = self.rng.integers(0, len(self.keys))
            return self.keys[index]

    def __setitem__(self, key, stepids):
        with self.lock:
            self.indices[key] = len(self.keys)
            self.keys.append(key)

    def __delitem__(self, key):
        with self.lock:
            idx = self.indices.pop(key)
            last = self.keys.pop()
            if idx != len(self.keys):
                self.keys[idx] = last
                self.indices[last] = idx

    def prioritize(self, stepids, priorities):
        # Uniform-Sampler ignoriert Prioritäten
        pass


class SampleTreeEntry:
    __slots__ = ('parent', 'key', 'uprob')
    def __init__(self, key=None, uprob=0.0):
        self.parent = None
        self.key = key
        self.uprob = uprob


class SampleTreeNode:
    __slots__ = ('parent', 'children', 'uprob')

    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.uprob = 0

    def __len__(self):
        return len(self.children)

    def append(self, child):
        # Falls child schon woanders hängt, dort entfernen:
        if hasattr(child, 'parent') and child.parent is not None:
            child.parent.remove(child)
        child.parent = self
        self.children.append(child)
        self.recompute()

    def remove(self, child):
        # Sucht child in children und ersetzt es ggf. durch das letzte
        idx = None
        for i, c in enumerate(self.children):
            if c is child:
                idx = i
                break
        if idx is None:
            return  # child nicht gefunden
        # "Swap and pop"
        last = self.children.pop()  # letztes Element
        if last is not child:
            self.children[idx] = last
            last.parent = self
        # neu berechnen
        self.recompute()

    def recompute(self):
        self.uprob = sum(c.uprob for c in self.children)
        if self.parent is not None:
            self.parent.recompute()


class SampleTree:
    """
    Einfacher Sum-Tree mit Knoten und Blättern, Array-loses Modell,
    aber OHNE "self.last"-Tricks, um Zyklen zu vermeiden.
    """
    def __init__(self, branching=16, seed=0):
        import numpy as np
        self.branching = branching
        self.root = SampleTreeNode()
        self.entries = {}  # key -> SampleTreeEntry
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.entries)

    def insert(self, key, uprob):
        entry = SampleTreeEntry(key, uprob)
        self.entries[key] = entry
        # Füge am "flachsten" Knoten an (z. B. root), oder
        # verteile auf Kinder, je nach branching-Logik...
        # Hier vereinfachen wir und hängen ALLE Blätter direkt an self.root an:
        self.root.append(entry)

    def remove(self, key):
        if key not in self.entries:
            return
        entry = self.entries.pop(key)
        if entry.parent:
            entry.parent.remove(entry)

    def update(self, key, uprob):
        if key not in self.entries:
            return
        entry = self.entries[key]
        entry.uprob = uprob
        if entry.parent:
            entry.parent.recompute()

    def sample(self):
        node = self.root
        # Gehe so lange in die Kinder, bis wir ein Leaf (SampleTreeEntry) finden
        while True:
            if not node.children:
                # kann nur passieren, wenn die root keine Kinder hat => leer
                return None
            # Sum der Kinder
            total = node.uprob
            if total <= 0:
                # Alle uprob=0 => wähle zufällig
                idx = self.rng.integers(0, len(node.children))
                chosen = node.children[idx]
            else:
                r = self.rng.random() * total
                s = 0
                for child in node.children:
                    s += child.uprob
                    if s >= r:
                        chosen = child
                        break
            if isinstance(chosen, SampleTreeEntry):
                return chosen.key
            node = chosen  # Knoten -> gehe eine Ebene tiefer


class Prioritized:
    """
    Einfacher Prioritized-Sampler (Sum-Tree-ähnlich).
    """
    def __init__(self, exponent=1.0, initial=1.0, branching=16, seed=0):
        self.exponent = float(exponent)
        self.initial = float(initial)
        self.tree = SampleTree(branching, seed)
        self.prios = collections.defaultdict(lambda: self.initial)
        self.items = {}
        self.stepitems = collections.defaultdict(list)

    def __len__(self):
        return len(self.items)

    def __call__(self):
        return self.tree.sample()

    def __setitem__(self, key, stepids):
        self.items[key] = stepids
        for sid in stepids:
            self.stepitems[sid].append(key)
        self.tree.insert(key, self._aggregate(key))

    def __delitem__(self, key):
        self.tree.remove(key)
        stepids = self.items.pop(key, [])
        for sid in stepids:
            self.stepitems[sid].remove(key)
            if not self.stepitems[sid]:
                del self.stepitems[sid]
                del self.prios[sid]

    def prioritize(self, stepids, priorities):
        if len(stepids) != len(priorities):
            return
        for sid, p in zip(stepids, priorities):
            self.prios[sid.tobytes()] = p
        # Aktualisiere ggf. alle Items, die diese stepids beinhalten
        # (Hier kann man sich Optimierungen überlegen, aber wir machen es direkt.)
        changed_keys = set()
        for sid in stepids:
            sid_b = sid.tobytes()
            if sid_b in self.stepitems:
                changed_keys |= set(self.stepitems[sid_b])
        for key in changed_keys:
            self.tree.update(key, self._aggregate(key))

    def _aggregate(self, key):
        """Aggregiert die Prioritäten aller Steps eines Items."""
        stepids = self.items[key]
        prios = [self.prios[sid] for sid in stepids]
        # exponent könnte man anwenden => p^alpha
        return sum(prios) / len(prios)


# ---------------------------------------------------------------------
# 2) Die Replay-Klasse
# ---------------------------------------------------------------------

class Replay:
    """
    Replay Buffer, der in "Chunks" speichert. Jeder Chunk kann bis zu chunksize Steps aufnehmen.
    Wir verwalten Items, die Startpunkte von Episoden (oder Sub-Episoden) sind. So entsteht
    eine "Liste" von möglichen Startpunkten, aus denen wir wahlweise Sequenzen abgreifen.

    - Mit use_priority=True wird ein Prioritized-Sampler verwendet (andernfalls Uniform).
    - Wir vermeiden jetzt gezielt, dass ein leerer Chunk in den Sampler gelangt.
    """

    def __init__(self, length, capacity=None, directory=None,
                 chunksize=1024, online=False, name='unnamed', seed=0,
                 use_priority=True):
        """
        Parameter:
          length:     Erwartete Episodenlänge bzw. Sequenzlänge
          capacity:   Max. Anzahl an "Items" (Startpunkte) im Replay
          directory:  Falls angegeben, werden Chunks dorthin gespeichert/geladen
          chunksize:  Wie viele Steps pro Chunk
          online:     Ob "online"-Verhalten (ggf. anderes Enqueuing)
          seed:       Zufalls-Seed für Sampler
        """
        self.length = length
        self.capacity = capacity
        self.chunksize = chunksize
        self.name = name
        self.online = online
        self.use_priority = use_priority

        # Wähle Sampler:
        if use_priority:
            self.sampler = Prioritized(seed=seed)
        else:
            self.sampler = Uniform(seed=seed)

        # Dictionaries:
        self.chunks = {}   # chunkid -> Chunk
        self.refs = {}     # chunkid -> int (Referenzen)
        self.items = {}    # itemid -> (chunkid, index)
        self.fifo = deque()
        self.itemid = 0

        # Pro Worker ein "aktueller" Chunk und ein Stream:
        self.current = {}         # worker -> (chunkid, index)
        self.streams = defaultdict(deque)

        # Locks und Metriken
        self.refs_lock = threading.RLock()
        self.rwlock = RWLock()
        self.metrics = {'samples': 0, 'inserts': 0, 'updates': 0}

        # Optionales Online-Feature
        if online:
            self.lengths = defaultdict(int)
            self.queue = deque()

        # Directory / asynchrones Speichern
        self.directory = Path(directory) if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.saved = set()

    def __len__(self):
        return len(self.items)

    def stats(self):
        """Einfache Statistik."""
        chunk_nbytes = sum(c.nbytes for c in self.chunks.values() if c.length > 0)
        return {
            'items': len(self.items),
            'chunks': len(self.chunks),
            'ram_gb': chunk_nbytes / (1024**3),
        }

    def add(self, step, worker=0):
        """
        Fügt einen Transition-Step dem Replay hinzu.
        Falls done==1.0 oder die Stream-Länge >= self.length, wird ein neuer
        Startpunkt ins Replay-Item eingefügt.
        """
        # Entferne Log-Keys:
        step = {k: v for k, v in step.items() if not k.startswith('log/')}

        with self.rwlock.reading():
            step = {k: np.asarray(v) for k, v in step.items()}

            # Falls kein aktueller Chunk für diesen Worker, erstelle einen:
            if worker not in self.current:
                new_chunk = Chunk(self.chunksize)
                with self.refs_lock:
                    self.refs[new_chunk.uuid] = 1  # 1 Referenz
                self.chunks[new_chunk.uuid] = new_chunk
                self.current[worker] = (new_chunk.uuid, 0)

            chunkid, index = self.current[worker]
            chunk = self.chunks[chunkid]

            # Erzeuge stepid = chunkid + index
            stepid = np.frombuffer(chunkid + index.to_bytes(4, 'big'), dtype=np.uint8)
            step['stepid'] = stepid

            # An den Chunk anhängen
            chunk.append(step)

            # Update Stream
            stream = self.streams[worker]
            stream.append((chunkid, index))

            # Referenzen updaten
            with self.refs_lock:
                self.refs[chunkid] += 1

            # Index weiterzählen
            index += 1
            if index < chunk.size:
                # noch Platz im Chunk
                self.current[worker] = (chunkid, index)
            else:
                # Chunk ist voll -> neue "Nachfolgerchunk"
                self._complete(chunk, worker)

            # Falls Episode zu Ende oder Stream ausreichend lang -> Insert
            done_flag = step.get('done', 0.0)
            if done_flag == 1.0:
                # Spüle ganzen Stream
                while stream:
                    cid, idx = stream.popleft()
                    self._insert(cid, idx)
                    self.metrics['inserts'] += 1
            else:
                if len(stream) >= self.length:
                    cid, idx = stream.popleft()
                    self._insert(cid, idx)
                    self.metrics['inserts'] += 1

            # Optional: im Online-Modus ggf. Queue füllen
            if self.online:
                self.lengths[worker] += 1
                if self.lengths[worker] % self.length == 0:
                    # Einfaches Bsp: wir fügen den Startpunkt in self.queue
                    self.queue.append((chunkid, index - 1))

    def _complete(self, chunk, worker):
        """Schließt chunk ab und erstellt einen Nachfolger."""
        succ = Chunk(self.chunksize)
        with self.refs_lock:
            # Die aktuelle Referenz dieses Chunks reduzieren
            self.refs[chunk.uuid] -= 1
            # Der neue Chunk wird von 2 Referenzen gehalten:
            #  (1) self.current[worker]
            #  (2) Querverbindung via chunk.succ
            self.refs[succ.uuid] = 2

        self.chunks[succ.uuid] = succ
        self.current[worker] = (succ.uuid, 0)
        chunk.succ = succ.uuid

    def _insert(self, chunkid, index):
        """
        Legt ein neues "Item" (Startpunkt) im Replay an, sofern ab (chunkid,index)
        noch Daten vorhanden sind.
        """
        chunk = self.chunks[chunkid]
        # Verhindere, dass wir bei leeren oder invaliden Indizes inserten:
        if chunk.length <= index:
            # Keine Daten mehr in diesem Chunk ab index
            return

        itemid = self.itemid
        self.itemid += 1
        self.items[itemid] = (chunkid, index)
        self.fifo.append(itemid)

        # Falls capacity erreicht, ältestes Item rausschmeißen
        if self.capacity and len(self.items) > self.capacity:
            self._remove_oldest()

        # Um den Sampler zu initialisieren, holen wir eine Sequenz (notfalls kurz):
        seq = self._getseq(chunkid, index, concat=True)
        if seq is None or len(seq.get('stepid', [])) == 0:
            # Keine sinnvolle Sequenz -> nicht in Sampler
            self.items.pop(itemid, None)
            self.fifo.pop()
            return

        stepids = seq['stepid']
        # Sampler updaten
        self.sampler[itemid] = [sid.tobytes() for sid in stepids]

    def _remove_oldest(self):
        """Entfernt das älteste Item (FIFO) aus dem Replay."""
        oldest_id = self.fifo.popleft()
        self.sampler.__delitem__(oldest_id)
        chunkid, _ = self.items.pop(oldest_id)
        with self.refs_lock:
            self.refs[chunkid] -= 1
            if self.refs[chunkid] < 1:
                # Chunk entfernen
                del self.refs[chunkid]
                old_chunk = self.chunks.pop(chunkid)
                # Auch den Nachfolger-Chunk 1 Referenz wegnehmen
                succid = old_chunk.succ
                if succid in self.refs:
                    self.refs[succid] -= 1

    def _getseq(self, chunkid, index, concat=True):
        """
        Sammelt bis zu self.length Schritte ab (chunkid, index) über chunk-Grenzen hinweg,
        bis done==1.0 oder self.length erreicht ist. Gibt Dict {key: np.array(...)} zurück
        oder None, wenn nichts gefunden.
        """
        steps = []
        cur_chunkid = chunkid
        cur_index = index
        needed = self.length
        while needed > 0 and cur_chunkid in self.chunks:
            chunk = self.chunks[cur_chunkid]
            # Wieviele Steps sind in diesem Chunk ab cur_index verfügbar?
            av = chunk.length - cur_index
            if av <= 0:
                # Hier ist nichts mehr -> break
                break
            use = min(av, needed)
            sliced = chunk.slice(cur_index, use)
            if not sliced:
                break  # slice() hat nichts geliefert (z.B. data=None)
            # Wir gehen Step für Step durch und prüfen done
            done_indices = None
            for i in range(sliced['stepid'].shape[0]):
                # Single Step extrahieren
                single_step = {k: sliced[k][i] for k in sliced}
                steps.append(single_step)
                needed -= 1
                if 'done' in single_step and single_step['done'] == 1.0:
                    done_indices = i  # markiere das
                    break

            if done_indices is not None:
                # wir haben done gefunden -> Sequenzende
                break

            cur_index += use
            # Falls wir noch weitere Steps brauchen, wechseln wir in den Nachfolgerchunk
            if chunk.succ and chunk.succ in self.chunks:
                cur_chunkid = chunk.succ
                cur_index = 0
            else:
                # kein Nachfolger oder unbekannt
                break

        if len(steps) == 0:
            return None

        # Zu Arrays zusammenbauen (concat)
        keys = steps[0].keys()
        if concat:
            out = {}
            for k in keys:
                out[k] = np.stack([step[k] for step in steps], axis=0)
            return out
        else:
            # Return Liste von Listen
            # (wurde an manchen Stellen nicht mehr benötigt, da wir concat==True wollen)
            out = {}
            for k in keys:
                out[k] = [step[k] for step in steps]
            return out

    def sample_subsequence(self, max_seq_len=None):
        """
        Samplet eine Sequenz ab einem beliebigen Replay-Item. Wir ziehen (chunkid,index),
        lesen ab dort, bis done==1.0 oder max_seq_len erreicht ist.
        Gibt (seq_as_list_of_dict, weight) zurück oder (None, None).
        """
        if len(self.items) == 0:
            return None, None

        for _ in range(10):
            try:
                itemid = self.sampler()
                chunkid, index = self.items[itemid]
                # Bilde Sequenz
                steps = []
                cur_chunkid = chunkid
                cur_index = index
                length_collected = 0
                while True:
                    if cur_chunkid not in self.chunks:
                        break
                    c = self.chunks[cur_chunkid]
                    av = c.length - cur_index
                    if av <= 0:
                        break
                    to_take = av if max_seq_len is None else min(av, max_seq_len - length_collected)
                    part = c.slice(cur_index, to_take)
                    if not part:
                        break
                    for i in range(part['stepid'].shape[0]):
                        single = {k: part[k][i] for k in part}
                        steps.append(single)
                        length_collected += 1
                        if max_seq_len is not None and length_collected >= max_seq_len:
                            break
                        if single.get('done', 0.0) == 1.0:
                            break
                    if max_seq_len is not None and length_collected >= max_seq_len:
                        break
                    if any(s.get('done', 0.0) == 1.0 for s in steps[-1:]):
                        # letztes Step done -> break
                        break
                    # Weiter in nächsten Chunk
                    if c.succ and c.succ in self.chunks:
                        cur_chunkid = c.succ
                        cur_index = 0
                    else:
                        break
                if steps:
                    weight = 1.0  # Bsp. Weighted-IS oder so
                    return steps, weight
            except KeyError:
                continue
        return None, None

    def update(self, data):
        """
        Aktualisiert Prioritäten und/oder Felder. data z. B.:
          data = {'stepid': ..., 'priority': ...}
        stepid-Shape [B, T, StepID-Dim], priority-Shape [B, T].
        """
        stepid = data.pop('stepid', None)
        priority = data.pop('priority', None)
        if stepid is None:
            return
        # shape: (B, T, XY) => flatten
        B, T, IDdim = stepid.shape
        stepid_2d = stepid.reshape([-1, IDdim])
        if priority is not None:
            # (B, T) => flatten
            priority_1d = priority.reshape([-1])
            self.sampler.prioritize(stepid_2d, priority_1d)
        # Falls wir Einträge direkt in den Chunks ändern wollen:
        # data enthält dann Key->Value (B, T, ?)
        # Hier könnte man implementieren: self._setseq(...)
        # Wir lassen das fürs Beispiel weg.

    def save(self):
        """Speichert alle Chunks (nur einmal)."""
        if not self.directory:
            return
        with self.rwlock.writing():
            # Schließe erst alle "halboffenen" Chunks
            for worker, (cid, idx) in self.current.items():
                c = self.chunks[cid]
                if c.length > 0:
                    self._complete(c, worker)

            # Speicher alle
            for cid, chunk in self.chunks.items():
                if chunk.length > 0 and cid not in self.saved:
                    chunk.save(self.directory)
                    self.saved.add(cid)

    @elements.timer.section('replay_load')
    def load(self, directory=None, amount=None):
        directory = directory or self.directory
        amount = amount or self.capacity or np.inf
        if not directory:
            return
        filenames = list(elements.Path(directory).glob('*.npz'))
        for fname in filenames:
            chunk = Chunk.load(fname, error='none')
            if chunk is not None:
                self.chunks[chunk.uuid] = chunk
        numitems = self._numitems(list(self.chunks.values()))
        for chunk in self.chunks.values():
            for index in range(numitems.get(chunk.uuid, 0)):
                self._insert(chunk.uuid, index)

    @elements.timer.section('complete_chunk')
    def _complete(self, chunk, worker):
        succ = Chunk(self.chunksize)
        with self.refs_lock:
            self.refs[chunk.uuid] -= 1
            self.refs[succ.uuid] = 2
        self.chunks[succ.uuid] = succ
        self.current[worker] = (succ.uuid, 0)
        chunk.succ = succ.uuid
        #print(f"[DEBUG Replay] Completed chunk {elements.UUID(chunk.uuid).hex}; created successor {elements.UUID(succ.uuid).hex}")
        return succ

    def _numitems(self, chunks):
        numitems = {}
        for chunk in chunks:
            if hasattr(chunk, 'length'):
                numitems[chunk.uuid] = chunk.length
        return numitems

    def _notempty(self, reason=False):
        if reason:
            return (True, 'ok') if len(self.sampler) > 0 else (False, 'empty buffer')
        else:
            return bool(len(self.sampler))


# ---------------- DummyDataset als Wrapper für DataLoader ------------------
class DummyDataset:
    def __init__(self, replay_memory):
        self.replay_memory = replay_memory
        self.seq_len = replay_memory.length
    def __len__(self):
        return max(1, len(self.replay_memory))
    def __getitem__(self, idx):
        data = self.replay_memory.sample(1, mode='train')
        if data is None:
            return {'reward': np.zeros((self.seq_len, 1), dtype=np.float32)}
        return data

# ---------------- Testlauf ------------------
if __name__ == '__main__':
    # Parameter für den Replay Buffer
    seq_len = 50
    buffer_capacity = 100000
    replay = Replay(length=seq_len, capacity=buffer_capacity, directory="replay_dir",
                    chunksize=1024, online=False, use_priority=True,
                    name='test_replay', seed=42)
    # Simuliere das Hinzufügen von Transitionen in zwei Episoden
    for episode in range(2):
        ep_length = np.random.randint(30, 70)
        for t in range(ep_length):
            done = 1.0 if t == ep_length - 1 else 0.0
            step = {
                'reward': np.random.uniform(-1, 1),
                'done': done,
                'obs': np.random.rand(20),
                'action': np.random.rand(4)
            }
            replay.add(step, worker=0)
        time.sleep(0.05)
    print("[DEBUG Replay] Stats:", replay.stats())
    dataset = DummyDataset(replay)
    for i in range(3):
        batch = dataset[i]
        print(f"[DEBUG Replay] Sampled batch {i}:")
        for key, value in batch.items():
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")



