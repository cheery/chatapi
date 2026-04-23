# ChatAPI

This specification describes a server hosting and connecting to LLM agents.

Without a layer of some kind, agents you invoke, require
at least their own API keys to run.
For convenience you usually give them the whole keyring.
If something goes awry and the agent leaks the keys,
it requires you to disable all of them.

Every computer running agents could have
a local server that hosts local agents and holds keyrings
that allow an access to remote agents.
In addition to providing security, it would keep the
interface uniform between different agents.

Agent server would be connected via a TCP or unix socket.
It would use chatapi to communicate.

## Low level details:

VLQ value in the beginning of the message is big-endian that
has the last high bit cleared.

The VLQ is encoded as follows:

    def encode_vlq(n):
        vlq = bytearray()
        vlq.append(n & 127)
        while n > 127:
            n >>= 7
            vlq.append(128 | (n & 127))
        vlq.reverse()
        return vlq

A message could be encoded like this:

    def encode_message(message):
        out = encode_vlq(len(message))
        out.extend(message)
        return out

Message is structured as follows:

    message := (arg RS)* arg GS payload

    arg is utf-8 text, not containing < 0x20 -characters.
    payload can contain any byte.

    GS = 0x1d
    RS = 0x1e

Every message is logically presented in python call format.
The call includes the payload in bytes
and the first arg is the name of the call.

Note that these messages resemble chatfmt, but they're not.
The Chatfmt comes along as the payload.

## Structure of messages

Message IDs: The participant that connected can produce even message IDs when it instantiates a message.
             The server produces odd message IDs when it instantiates a message.

The first arg is the name of the call, and the name describes a flavor of the message.
The second arg is the message ID encoded as text integer.

There are several suffixes to the messages, translated to their python-counterparts:

* `*?` == `_stream_request`
* `?` == `_request`
* `*!` == `_stream_response`
* `!` == `_response`

There's a version handshake.

    version?(0, my_version)

The server responds with one of these:

    version!(0)
    not_supported!(0, highest_supported_version)

If the server supports the version, it should respond with it, and from that on use the given version.
This version that you're reading right now is labeled as "0".

Client may query what is supported by:

   supported?(0, "")

And the server responds with multiple responses for each supported version.
From earliest to latest.

   supported*!(0, X)
   supported*!(0, Y)
   supported!(0, Z)

The message format will grow and change as the first version is implemented.

## Termination of stream

Both client and server may terminate the stream by writing:

  bye(reason)

The bye, as it is the last message, doesn't need a message ID.
client doesn't need to describe the reason for quitting (it can be ""), it is assumed that it was a normal quit.

## Authentication

The protocol assumes that when the user can access the protocol,
they are allowed to call and access bots behind it.
This may change in future, but the authentication
stays lightweight as this is meant to be a local protocol.

## Message format

chatfmt.md is used as the format for autoregressive large language models.
Examples of interactions:

    client: chat?(2, "")
    server: chat!(2, X) # X indicates an ID given to the chat session.

    client: message*?(4, X, chatfmt formatted request continues)
    client: message*?(4, X, chatfmt formatted request continues)
    client: message?(4, X, chatfmt formatted request ended)

    server: message*!(4, X, chatfmt formatted response continues)
    server: message*!(4, X, chatfmt formatted response continues)
    server: message!(4, X, chatfmt formatted response ended)

    client: message?(6, X, chatfmt formatted request)
    server: message*!(6, X, chatfmt formatted response continues)
    server: message*!(6, X, chatfmt formatted response continues)
    client: abort?(8, X)
    server: abort!(8, X)
    server: aborted!(6, X)

    client: end?(8, X)
    server: end!(8, "")

Server may also respond with `ending(message_id, X, reason)` at any time.

This protocol is prospective as long as first implementation is not implemented.
