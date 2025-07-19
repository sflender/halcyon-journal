Build a local-only interactive journaling app in Python with the following features:

- When launched, there are two options for the user: "journal" and "reflect"
- journaling mode starts a new (daily) journal entry for the user to write about their day. when the user is done with daily journaling, respond to the user like a personal therapist, using context from the existing journal. also pull up a few of related journaling entries for the user to browse using some sort of similary search.
- each daily journal entry is saved as a plain Markdown file locally to preserve the user's privacy.
- reflect mode lets the user prompt their own journal directly and ask questions such as "how have I been doing?"
- all processing must be local to preserve privacy, so use a small local LLM such as OLama