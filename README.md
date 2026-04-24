# ChatAPI

This project describes a protocol, a chat format
and implements a simple server (chatapi-server),
and an agent (chatapi-agent).

The idea is that without a layer of some kind,
agents you invoke require their own API keys to run.
For convenience you usually give them the whole ring of keys.
If something goes awry and the agent leaks the keys,
it requires you to disable all of your keys on different
providers.

ChatAPI solves two problems: The difficulty of switching
a provider because it talks different chat format,
and the above problem of passing keys in.
ChatAPI provides an uniform format for LLM logs.
It implements a local server that is supposed to be able to
run local models in future,
but also connect to remote locations to serve models.

As of now, the format provides tool call support but
does not support multimodal communication.

chatapi-server is a server that runs either as a service
as its own user or in a container. It has access to API keys.

chatapi-agent is a simple pi-agent -inspired agent,
with ability to run bash commands,
that runs on any safe nono.sh env or a container.

License: MIT
