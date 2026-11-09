//! The kinds of subagents: what each is for, its instructions and its tools.

use crate::tools::Tool;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    General,
    Explore,
    Implement,
    Verify,
    Review,
    Formalize,
    Document,
}

const READ: [Tool; 4] = [Tool::ReadFile, Tool::Grep, Tool::Glob, Tool::ListDir];

impl Role {
    pub const ALL: [Role; 7] = [
        Role::General,
        Role::Explore,
        Role::Implement,
        Role::Verify,
        Role::Review,
        Role::Formalize,
        Role::Document,
    ];

    /// The name the model gives it
    pub fn name(self) -> &'static str {
        match self {
            Role::General => "general",
            Role::Explore => "explore",
            Role::Implement => "implement",
            Role::Verify => "verify",
            Role::Review => "review",
            Role::Formalize => "formalize",
            Role::Document => "document",
        }
    }

    pub fn from_name(name: &str) -> Option<Role> {
        let name = name.trim().to_ascii_lowercase();
        Role::ALL.into_iter().find(|r| r.name() == name)
    }

    /// As the transcript shows its tasks
    pub fn label(self) -> &'static str {
        match self {
            Role::General => "Task",
            Role::Explore => "Explore",
            Role::Implement => "Implement",
            Role::Verify => "Verify",
            Role::Review => "Review",
            Role::Formalize => "Formalize",
            Role::Document => "Document",
        }
    }

    /// The tools it can call
    pub fn tools(self) -> Vec<Tool> {
        let edit = [Tool::EditFile, Tool::WriteFile];
        let mut tools = READ.to_vec();
        match self {
            Role::Explore => {}
            Role::Verify | Role::Review => tools.push(Tool::Bash),
            Role::Document => tools.extend(edit),
            Role::General | Role::Implement | Role::Formalize => {
                tools.extend(edit);
                tools.push(Tool::Bash);
            }
        }
        tools
    }

    /// What it does, in its system prompt
    pub fn instructions(self) -> &'static str {
        match self {
            Role::General => {
                "\
Do the task with the tools. Change files only if the task asks you to, and check the change \
(build, tests) when you can."
            }
            Role::Explore => {
                "\
You investigate the code to answer the question of the task. You can't change anything. Search \
before reading, and read the relevant parts of large files. Answer precisely: the files, types \
and functions involved, with paths and line numbers, how they connect, and what you could not \
determine."
            }
            Role::Implement => {
                "\
You make the change described in the task, and only that change. Read the code you change and \
what it depends on first; keep to the style of the surrounding code. Then build and run the \
relevant tests, and fix what fails. Don't commit. Report the files changed with what changed in \
each, the commands you ran and their results, and anything left undone or that you are unsure \
of."
            }
            Role::Verify => {
                "\
You check, independently and skeptically, whether the change or claim described in the task is \
correct and complete. Don't trust the description: build, run the tests, run the program, look \
at the code. You may write small checks, but only in a temporary directory (mktemp -d), never in \
the project. Don't fix anything. Start your report with the verdict, PASS or FAIL, then the \
evidence: the commands run with the relevant output, and for a FAIL what is wrong, with paths \
and line numbers."
            }
            Role::Review => {
                "\
You review the changes described in the task (see git diff and git status, or the files given) \
for bugs, missed edge cases, error handling, unclear code, and differences from what was asked. \
Don't change anything. Report your findings from the most to the least severe, each with \
path:line, the problem, why it matters and a fix; say plainly if you found nothing that \
matters."
            }
            Role::Formalize => {
                "\
You make the requirements or behavior described in the task precise: state the invariants, \
preconditions and postconditions, the edge cases and examples of inputs with their expected \
outputs. Where the project has tests, encode them as tests (property-based if the project \
already uses it) and run them. Change only test and specification files, never the code under \
test. Report the specification, the tests added with their results (a failing test is a \
finding, not something to fix), and any ambiguity in the requirements."
            }
            Role::Document => {
                "\
You write or update the documentation described in the task: README, guides, doc comments. \
Read the code so that what you write is accurate, and keep to the style of the existing \
documentation. Don't change what the code does. Report the files changed with a short summary \
of each."
            }
        }
    }

    /// It can edit or write files: its tasks run alone, never at the same
    /// time as others
    pub fn changes_files(self) -> bool {
        self.tools()
            .iter()
            .any(|t| matches!(t, Tool::EditFile | Tool::WriteFile))
    }

    /// Tool calls for one task at most
    pub fn max_steps(self) -> usize {
        match self {
            Role::Explore | Role::Review | Role::Document => 40,
            _ => 60,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_tools() {
        for r in Role::ALL {
            assert_eq!(Role::from_name(r.name()), Some(r));
            assert!(!r.tools().contains(&Tool::Task));
        }
        // the roles in the task tool's definition are these
        let def = Tool::Task.definition();
        for r in Role::ALL {
            assert!(def.contains(&format!("\"{}\"", r.name())), "{}", r.name());
        }
        assert!(!Role::Explore.tools().contains(&Tool::EditFile));
        assert!(!Role::Verify.tools().contains(&Tool::WriteFile));
        assert!(Role::Implement.tools().contains(&Tool::Bash));
        let alone: Vec<_> = Role::ALL
            .into_iter()
            .filter(|r| r.changes_files())
            .collect();
        assert_eq!(
            alone,
            [
                Role::General,
                Role::Implement,
                Role::Formalize,
                Role::Document
            ]
        );
    }
}
