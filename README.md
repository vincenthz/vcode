# vcode

A terminal coding agent in the spirit of Claude Code, Codex or opencode, with
the model running on your machine: Qwen GGUFs on [biv](https://github.com/vincenthz/biv)
(pure Rust inference, Metal or CPU). No server, no API key, nothing leaves the
machine.

```sh
cargo build --release
./target/release/vcode                 # in the current directory
./target/release/vcode ~/src/project   # or another one
./target/release/vcode -m hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_1
./target/release/vcode -y -p "the tests fail, fix them"   # one task, no interface
./target/release/vcode --pilot         # the agent plans and directs subagents
```

Models come from the Hugging Face cache, named as biv names them. Without `-m`,
vcode takes Qwen3.6 35B-A3B (Q8_0) if it is downloaded, else Qwen3.8 27B
(Q4_1):

```sh
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-Q8_0.gguf   # 37.8 GB
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_1.gguf             # 17.5 GB
```

The mixture of experts suits an agent better: an agent's turn is many short
prompts (tool results) and answers, and on an M2 Max the 35B-A3B processes
prompts at ~750 tok/s and answers at ~100 tok/s, against ~170 and ~35 for the
27B. Fixing two bugs in a small crate (run the tests, read, edit twice, run
the tests again) takes 12 s with it, 35 s with the 27B.

## Tools

| tool | does | asks first |
|---|---|---|
| `read_file` | lines of a file with their numbers, 400 at a time | no |
| `grep` | regex search, skipping what git ignores | no |
| `glob` | files by pattern, newest first | no |
| `list_dir` | entries of a directory | no |
| `edit_file` | exact replacement of a unique string, shown as a diff | yes |
| `write_file` | create or overwrite a file, shown as a diff | yes |
| `bash` | a command in the working directory, with a timeout | yes |
| `task` | gives a task to a subagent, see below | no (its calls do) |
| `update_plan` | the pilot's view of the project and its task list | no |

The permission mode (shift+tab) goes from *ask before edits and commands* to
*accept edits* to *accept everything*; `-y` starts in the last one. When
asked: `y` allows, `a` allows and stops asking for that kind of call, `n`
refuses (the model is told and goes on), `esc` refuses and stops the turn.

The system prompt holds the working directory, platform, date, git branch and
top-level entries, and the project's `AGENTS.md` (or `VCODE.md`, `CLAUDE.md`).
It is processed once at startup and saved, so `/clear` is instant.

## Subagents

`task(agent, description, prompt)` hands work to a subagent, which starts
knowing nothing of the conversation, works in a context of its own
(`--sub-ctx`, 32k by default; `0` turns subagents off), runs its own tool loop,
and whose final report is the result of the call. The main conversation only
grows by that report, however many files the subagent read. Its edits and
commands are approved like the agent's, and its calls show under the task,
collapsed once it is done (ctrl+o shows them).

Each kind of subagent has its own instructions and tools:

| agent | tools | does |
|---|---|---|
| `explore` | read, search | investigates the code and answers questions; changes nothing |
| `implement` | all | makes one change, then builds and tests it |
| `verify` | read, search, bash | checks a change or a claim independently; reports PASS or FAIL with evidence |
| `review` | read, search, bash | reviews changes for bugs and differences from what was asked; changes nothing |
| `formalize` | read, search, edit, bash | makes requirements precise (invariants, edge cases, examples) and encodes them as tests |
| `document` | read, search, edit | writes or updates documentation |
| `general` | all | anything else |

They share one chat on the engine: the system prompt of each kind is processed
the first time it is used and saved, and every task restores its kind's (the
session is 1.5 GB with the 35B-A3B at 32k; a saved system prompt is a copy of
the recurrent states plus its tokens' KV cache). biv runs one sequence at a
time on the GPU, so tasks run one after the other.

Outside pilot mode the agent has every tool and delegates when it chooses to.
Qwen3.6 35B-A3B does when asked ("use a subagent to…"), but seldom on its own:
given an open question over several files, it reads them itself.

## Pilot mode

With `--pilot` (or `/pilot`, which starts a new conversation), the main agent
has no tools but `task` and `update_plan`: it can't read or change files, so
everything goes through subagents, and its context holds only the overall
view: the request, its overview of the project, the plan, and the reports. Its
instructions are to explore first, record a plan, then take the tasks one at a
time: implement, verify (and review when it matters), a new implement task
with what failed, formalize when the requirements are subtle, document when
users or interfaces change, and a summary at the end. A change is done when a
verify task passed.

The plan shows in the transcript (✓ done, ◐ in progress, ○ to do, ✗ blocked)
and its progress in the status line, and it is carried over when the
conversation is compacted.

## Interface

- enter sends; shift+enter, alt+enter, ctrl+j, or `\` then enter, start a new line
- esc interrupts the model or a running command; messages typed while it works are queued
- ctrl+t shows the reasoning, ctrl+o expands tool output and diffs
- pgup/pgdn or the mouse wheel scroll (`--no-mouse` leaves the mouse to the terminal, for selecting text)
- up/down go through the history
- `!cmd` runs a shell command; its output goes to the model with your next message

Commands: `/clear`, `/compact` (replace the conversation with the model's
summary of it, which also happens by itself at 80% of the context), `/pilot`,
`/think`, `/nothink`, `/mode`, `/context`, `/help`, `/quit`.

## Layout

| file | role |
|---|---|
| `src/main.rs` | options, the headless mode (`-p`) |
| `src/agent.rs` | the agent thread: owns the model and the chats (main, subagent), runs the tool loops, compaction |
| `src/roles.rs` | the kinds of subagents: instructions, tools |
| `src/tools.rs` | tool definitions (as the chat template's `tojson` writes them), planning (argument checks, diffs) and running |
| `src/prompt.rs` | the system prompts: agent, pilot, subagents |
| `src/ui/mod.rs` | the ratatui interface: transcript, approvals, status |
| `src/ui/text.rs` | word wrapping, a little Markdown |
| `src/ui/input.rs` | the multi-line input with history |

biv's engine is not `Send`, so the agent thread loads the model itself and
the two threads only exchange messages: `Cmd` to the agent, `Msg` back, and a
reply channel in each approval request.
