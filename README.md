# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/ajidayubi/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/ajidayubi/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions and Anthropic Messages APIs over ten free-tier model providers, routing each request to the best free model that can serve it and failing over when a provider refuses. It reads each provider's own model catalog, ranks free models by the brief's quality table, translates requests between the OpenAI, Anthropic and Gemini wires without dropping a field, and keeps answering while providers run dry. The same code runs against the real services when the brief (or the provider key environment variables) carries real keys and base URLs.

## Run it

```sh
python3 main.py --list
MAKEFUN_ENV_URL=http://127.0.0.1:8080 MAKEFUN_PROMPT_FILE=prompt.txt python3 main.py
```

`python3 main.py --list` prints the catalog document the server registers, and exits 0. In a trial the server registers its port and catalog with the env, serves the workload, and exits 0 when the env says the workload is done. Python 3.13 standard library only — nothing to install.

---

<sub>Entered on [make.fun](https://make.fun/?ref=cbk8hpf3&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
