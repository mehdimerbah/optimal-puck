RL Competition Hockey Agent
===========================

This is a simple example implementation for a client of the comprl hockey game.  It
wraps an agent from the `hockey` package and implements a script to run it as a client
that connects to the comprl server.
To run the example agent, you first need to install the dependencies:
```
pip install -r ./requirements.txt
```

Then execute the script `run_client.py` to run the client:
```
python3 ./run_client.py --server-url <URL> --server-port <PORT> \
    --token <YOUR ACCESS TOKEN> \
    --args --agent=strong
```

The server information can also be provided via environment variables, then they don't
need to be provided via the command line:
```
# put this in your .bashrc or some other file that is sourced before running the agent
export COMPRL_SERVER_URL=https://comprl.cs.uni-tuebingen.de
export COMPRL_SERVER_PORT=80
export COMPRL_ACCESS_TOKEN=08367c3d-fa07-4b2b-8f46-38fdd9d6055f
Then just call
```
python3 ./run_client.p --args --agent=strong
```
