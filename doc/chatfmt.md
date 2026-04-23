Autoregressive encodings will be around for a while.
I propose a format for storing message logs,
such that the log itself would be, in theory, producible by
autoregressive LLM itself.
Though some LLMs may not be able to do so because of their
tokenizing schemes.

Chatfmt is separated into sequence of messages.
Logically chatfmt message forms a python tag + args/kwargs bundle.
Allowing following kind of field types: text, bytes, int, float, void, boolean.
Fields starting with _ are meta -fields.

Valid tag name or field name match a regex: `[a-zA-Z_][a-zA-Z_0-9]*`

## Encoding

The bundle is first split into sections:

 1. tag contains the tag of the message.
 2. args contains the args of the message.
 3. kwargs contains the kwargs not fitting into other fields.
 4. body contains the "content" -field.
 5. meta contains the _ -prefixed fields.

The sections are encoded as follows:

    file := (msg FS)*
    msg := tag value* field* (GS body meta*)?
    field := US keyword value
    meta := US keyword value
    value := RS  text
           | DC1 text
           | DC2 text
           | DC3 blob
           | DC4
           | DC4 "true"
           | DC4 "false"

    FS, GS, US, RS, DC1, DC2, DC3, DC4 are corresponding ascii control characters.

    tag, keyword, text, body, are UTF-8 strings

    body uses \nn -escaping syntax for <0x20 -characters and backslash itself, excluding \r\n\t
    That is. \nn syntax is used for everything else except \r\n\t that pass as-it. Backslash itself is escaped as \5c.

    blob is a sequence of bytes preceded by it's length in vlq format (as described in chatapi.md:low level details)

    whenever text is preceded by DC1, it's interpreted as integer
    whenever text is preceded by DC2, it's interpreted as float
    DC4 is translated to python Ellipsis, it implies that the field is merely present.
    DC4 "null" is translated to null, also absense of field is translated to null.
    DC4 "true" is translated to boolean True
    DC4 "false" is translated to boolean False

    meta keyword fields are encoded without _ in the beginning.

The special character codes are listed below:

    EXCLUDE = 0x9, 0xA, 0xD
    DC1 = 0x11
    DC2 = 0x12
    DC3 = 0x13
    DC4 = 0x14
    FS  = 0x1c
    GS  = 0x1d
    RS  = 0x1e
    US  = 0x1f
    ESC = 0x5c

## Messages

Messages are written in python call format for convenience.

### System prompt

The system prompt starts a message sequence and describes
context to the assistant.

    system(content:str)

### Assistant message

Assistant message is a message printed to the user as it.

    assistant(content:str)

### Think -message

Think message it though process of the bot, it can be printed
of not printed to the user.

    think(content:str)

### User message

User message is user's response:

    user(content:str)

### Footnote

Message vocabulary is still too thin for real agent work.
But as we develop the first implementation, we will grow it.

