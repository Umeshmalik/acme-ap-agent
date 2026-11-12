"""The agent itself, independent of the terminal.

    core.py           Agent: the model/tool loop, approvals, stop and crash repair
    tools.py          the tool registry the model sees
    posting_rules.py  the checks that decide whether an invoice may be posted
    data.py           reads invoice files and vendors.json
    store.py          saves conversations and posts in SQLite
"""
