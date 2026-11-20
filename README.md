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
(`--sub-ctx`, 128k by default; `0` turns subagents off), runs its own tool loop,
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

The system prompt of each kind is processed the first time it is used and
saved, and every task gets a chat of its own that starts from its kind's (a
saved system prompt is a copy of the recurrent states plus its tokens' KV
cache; a task's session is about 3.5 GB with the 35B-A3B at 128k).

### Tasks at the same time

Tasks the model calls together, in one answer, run at the same time when their
kinds don't change files (`explore`, `verify`, `review`): up to `--parallel` of
them (4 by default), each subagent on a thread of its own. The engine is then
served by biv's [continuous batching](https://github.com/vincenthz/biv#batching)
scheduler, whose passes take the tokens of every chat ready for one, so the
weights are read once for all of them. Tasks that can change files
(`implement`, `formalize`, `document`, `general`) run alone, so two subagents
never edit the tree at the same time. The calls of each task show under it,
and approvals they ask for at the same time are asked one after the other.

Two explore tasks on the 35B-A3B (M2 Max) answer at about 50 tok/s each, about
100 tok/s together against 115 for one alone. Each task running needs its own
session: 4 at 128k are about 14 GB on top of the model. MTP drafts gain less in a
batch (see biv's numbers): with many tasks at once, `-d 1` can be faster.
`--parallel 1` runs tasks one after the other on the engine itself, without the
scheduler.

Outside pilot mode the agent has every tool and delegates when it chooses to,
and is told that explore, verify and review tasks called together run at the
same time.
Qwen3.6 35B-A3B does when asked ("use a subagent to…"), but seldom on its own:
given an open question over several files, it reads them itself.

## Pilot mode

With `--pilot` (or `/pilot`, which starts a new conversation), the main agent
has no tools but `task` and `update_plan`: it can't read or change files, so
everything goes through subagents, and its context holds only the overall
view: the request, its overview of the project, the plan, and the reports. Its
instructions are to explore first, record a plan, then take the tasks one at a
time: implement, verify (and review when it matters: the two together, at
the same time), a new implement task with what failed, formalize when the requirements are subtle, document when
users or interfaces change, and a summary at the end. A change is done when a
verify task passed.

The plan shows in the transcript (✓ done, ◐ in progress, ○ to do, ✗ blocked)
and its progress in the status line, and it is carried over when the
conversation is compacted.

## Sandbox

With `--sandbox`, every tool runs on another machine, in a work directory that
holds a copy of the project: the files read, searched and edited are those
there, and commands run there. The copy is made at the start, and kept between
sessions: the next start copies only what changed here since.

```sh
vcode --sandbox ssh:devbox                    # a host, in ~/vcode/<project>
vcode --sandbox ssh:me@host:2222,dir=/srv/w   # another port, another directory
vcode --sandbox vemu:dev,image=alpine.raw,kernel=Image   # a VM, made the first time
vcode --sandbox vemu:dev                      # and started again afterwards
```

A `vemu:` sandbox is a VM run by [vemu](https://github.com/vincenthz/vemu), on
a golden image from `vemu build` (with ssh for your key, and git) and the
kernel it was built with. Made on first use, its overlay and settings
(`memory=`, `cpus=`, `key=`, `append=`) are kept in
`~/.local/share/vcode/vms/NAME`, and it is booted for each session, with a free
port of this host forwarded to its ssh server, and powered off at the end. Its
console goes to `console.log` there. Nothing asks for approval in a VM (shift+tab
still changes that); on an ssh host, approvals are as without a sandbox. The
agent is told where it works, and that in a VM it may install what it needs.

What is copied: the files the project's `.gitignore` files don't ignore (so
not `target/`, nor an ignored `.env`), and the repository itself. Changes made
there come back only when you ask:

- `/sandbox` lists the files changed there since the copy
- `/sandbox diff` shows them, as diffs from the files here
- `/sandbox pull` brings them here; `/sandbox push` copies there what changed
  here since (both wait for the agent to be done)
- `-p … --pull` pulls once the task is done

A file changed on both sides is left as it is on both, and said so. Commits
made there stay there (pull brings the working tree only). `!cmd` runs in the
sandbox too; `!!cmd` runs here. On quitting, vcode says what was changed there
and not pulled: it is still there next time.

Everything goes through one ssh connection (ControlMaster), opened before the
interface starts, so a password or a passphrase can be asked for. The other
side needs only a POSIX shell, `find`, `tar` and `cksum` (busybox will do),
and uses `git` and `bash` when it has them. A command killed from the list
(ctrl+p) is killed there, with what it started.

## Interface

- enter sends; shift+enter, alt+enter, ctrl+j, or `\` then enter, start a new line
- esc interrupts the model or a running command; messages typed while it works are queued
- ctrl+t shows the reasoning, ctrl+o expands tool output and diffs
- pgup/pgdn or the mouse wheel scroll
- drag over the transcript to select, and the selection is copied when the
  button is released; double-click selects a word (or a path with its line
  number), triple-click a line; dragging past the top or bottom scrolls. The
  copy goes through `pbcopy` (`wl-copy`, `xclip`, `xsel` on Linux), or over
  ssh through the terminal (OSC 52; in tmux, `set -g set-clipboard on`).
  `--no-mouse` leaves the mouse to the terminal instead
- up/down go through the history
- `!cmd` runs a shell command (in the sandbox, if any; `!!cmd` here); its
  output goes to the model with your next message
- ctrl+p (or `/ps`) lists the shell commands running, the agents' and yours:
  ↑/↓ select, `k` kills one with everything it started, esc closes. A command
  whose shell exits but leaves processes running (`npm run dev &`) returns to
  the model and stays listed, in the background, until they end or are
  killed; the status line counts them. Quitting vcode kills them all
- the terminal's bell rings when a call waits for your approval, or an answer
  that took 10 s or more is done, while the terminal doesn't have the focus
  (or doesn't say whether it has it); most terminals then flag the tab or
  bounce the dock icon, tmux marks the window. `/bell off` or `--no-bell`
  silence it
- with `--notify` (or `/notify on`), a desktop notification says why at the
  same times: "waiting for your approval: cargo test", "done in 1m15s: All
  42 tests pass." It goes through the terminal when it shows them (iTerm2,
  kitty, WezTerm, Ghostty, foot: this also works over ssh; in tmux, `set -g
  allow-passthrough on`), else through `osascript` or `notify-send`

Commands: `/clear`, `/ps`, `/compact` (replace the conversation with the model's
summary of it, which also happens by itself at 80% of the context), `/pilot`,
`/think`, `/nothink`, `/mode`, `/context`, `/bell`, `/notify`, `/sandbox`, `/help`, `/quit`.

The context is 256k tokens by default (`-c`), at most the model's; a session
allocates its whole KV cache up front, about 6.5 GB with the 35B-A3B at 256k.
`/context` shows its use, and `/context 64k` changes its size while running:
the conversation goes on in a new session, its tokens processed again (it must
fit in 80% of the new size). `/context sub 64k` changes the subagents' for the
tasks that start from then on. Sizes are tokens, or with `k` or `m`.

## Layout

| file | role |
|---|---|
| `src/main.rs` | options, the headless mode (`-p`) |
| `src/agent.rs` | the agent thread: owns the model and the chats (main, subagents), the batch scheduler, runs the tool loops and the subagents' threads, compaction |
| `src/roles.rs` | the kinds of subagents: instructions, tools |
| `src/tools.rs` | tool definitions (as the chat template's `tojson` writes them), planning (argument checks, diffs) and running |
| `src/procs.rs` | the shell commands running, listed and killed from the interface |
| `src/sandbox.rs` | sandboxes: ssh hosts and vemu VMs, the tools' operations there, copying both ways |
| `src/tar.rs` | the tar archives the copies go in, and `cksum`'s CRC |
| `src/prompt.rs` | the system prompts: agent, pilot, subagents |
| `src/ui/mod.rs` | the ratatui interface: transcript, approvals, status |
| `src/ui/text.rs` | word wrapping, a little Markdown |
| `src/ui/input.rs` | the multi-line input with history |

biv's engine is not `Send`, so the agent thread loads the model itself and
the two threads only exchange messages: `Cmd` to the agent, `Msg` back, and a
reply channel in each approval request. With `--parallel` above 1 the agent
thread runs biv's batch scheduler, and the agent itself runs on a thread of
its own with the scheduler's handle, an engine that can be shared: the
subagents working at the same time make their chats on it from their threads.
