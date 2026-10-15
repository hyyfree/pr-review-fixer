/**
 * pr-review-fixer — a code agent that reviews a pull request and produces the
 * commit that fixes its most important finding.
 *
 * Input:  the PR URL (or `owner/repo#N`) anywhere in the request text.
 * Output: JSON with review findings, a unified-diff fix patch against the PR
 *         head, and the exact non-force push commands for the operator.
 *
 * Design notes:
 * - GitHub's REST API is called unauthenticated: public PRs need no token, so
 *   the agent never holds a secret it could leak or misuse.
 * - The review is one `respond()` call on the requested model (kimi-k2.5,
 *   gpt-5.4-nano as fallback) with the full unified diff; the model answers
 *   as a fenced JSON block, parsed leniently from the Responses payload.
 * - The platform cannot push: a code agent has no credentials by design. The
 *   agent emits `fix.patch` plus the exact `git` commands; the operator runs
 *   them with their own credentials. The commands never merge, never
 *   force-push, and only ever write to the reviewed PR's head branch.
 */

type AgentContext = {
    request: Request;
    model: (id: string) => unknown;
    respond: (settings: {
        model: unknown;
        instructions: string;
        prompt: string;
    }) => Promise<Response>;
};

export type PrRef = { owner: string; repo: string; number: number };

export type ReviewFinding = {
    severity: "critical" | "major" | "minor";
    file: string;
    title: string;
    detail: string;
};

export type ReviewReport = {
    summary: string;
    findings: ReviewFinding[];
    fix: { finding: string; patch: string; verify: string };
};

const PREFERRED_MODEL = "kimi-k2.5";
const FALLBACK_MODEL = "openai/gpt-5.4-nano";
/** Diffs beyond this are truncated: reviews stay focused and costs bounded. */
const MAX_DIFF_CHARS = 120_000;
const GITHUB_API = "https://api.github.com";

/**
 * Pull the PR reference out of the request text. Accepts a full
 * `https://github.com/owner/repo/pull/123` URL or the `owner/repo#123`
 * shorthand. Exported for tests.
 */
export function parsePrRef(text: string): PrRef | null {
    const url = text.match(/github\.com[/:]([\w.-]+)\/([\w.-]+)\/pull\/(\d+)/i);
    if (url) return { owner: url[1], repo: url[2], number: Number(url[3]) };
    const shorthand = text.match(/\b([\w.-]+\/[\w.-]+)#(\d+)\b/);
    if (shorthand) {
        const [owner, repo] = shorthand[1].split("/");
        return { owner, repo, number: Number(shorthand[2]) };
    }
    return null;
}

/**
 * Extract the last user message from either a Responses (`input`) or a Chat
 * Completions (`messages`) gateway body. Exported for tests.
 */
export function inputTextOf(body: unknown): string {
    if (!body || typeof body !== "object") return "";
    const record = body as Record<string, unknown>;
    if (typeof record.input === "string") return record.input;
    if (Array.isArray(record.input)) return partsText(record.input);
    if (Array.isArray(record.messages)) {
        for (let i = record.messages.length - 1; i >= 0; i--) {
            const message = record.messages[i] as Record<string, unknown>;
            if (message?.role === "user") return partsText([message.content]);
        }
    }
    return "";
}

function partsText(value: unknown): string {
    if (typeof value === "string") return value;
    if (Array.isArray(value)) return value.map(partsText).join(" ");
    if (value && typeof value === "object") {
        const part = value as Record<string, unknown>;
        return [part.text, part.input_text]
            .filter((t) => typeof t === "string")
            .join(" ");
    }
    return "";
}

/**
 * Tolerant extraction of the model's JSON answer: fenced block first, then
 * the first balanced `{...}` span. Returns null when nothing parses.
 * Exported for tests.
 */
export function extractReport(text: string): ReviewReport | null {
    const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
    const candidates = fenced ? [fenced[1], text] : [text];
    for (const candidate of candidates) {
        const start = candidate.indexOf("{");
        if (start < 0) continue;
        let depth = 0;
        let inString = false;
        let escaped = false;
        for (let i = start; i < candidate.length; i++) {
            const ch = candidate[i];
            if (escaped) {
                escaped = false;
                continue;
            }
            if (ch === "\\") {
                escaped = true;
                continue;
            }
            if (ch === '"') inString = !inString;
            if (inString) continue;
            if (ch === "{") depth++;
            if (ch === "}") depth--;
            if (depth === 0) {
                try {
                    const parsed = JSON.parse(candidate.slice(start, i + 1));
                    if (
                        parsed &&
                        typeof parsed.summary === "string" &&
                        Array.isArray(parsed.findings)
                    ) {
                        return parsed as ReviewReport;
                    }
                } catch {
                    /* keep scanning */
                }
                break;
            }
        }
    }
    return null;
}

async function github(path: string, accept: string): Promise<Response> {
    const response = await fetch(`${GITHUB_API}${path}`, {
        headers: {
            Accept: accept,
            "User-Agent": "pollinations-pr-review-fixer",
        },
    });
    if (!response.ok) {
        throw new Error(`GitHub ${path} failed: ${response.status}`);
    }
    return response;
}

export function capDiff(diff: string): { text: string; truncated: boolean } {
    if (diff.length <= MAX_DIFF_CHARS) return { text: diff, truncated: false };
    return { text: diff.slice(0, MAX_DIFF_CHARS), truncated: true };
}

/**
 * Extract the head branch name from a PR page's embedded JSON. Fallback for
 * when the REST API is rate-limited. Exported for tests.
 */
export function parseHeadRef(html: string): string | null {
    const match = html.match(/"headRefName":"([^"]+)"/);
    return match ? match[1] : null;
}

/**
 * Fetch PR metadata and the full unified diff. The diff comes from
 * github.com's pull/N.diff endpoint (served outside the API rate limit);
 * metadata prefers the REST API and degrades to the PR page's embedded
 * headRefName when the API throttles shared hosting IPs. The diff is capped;
 * the truncation is reported so the model can flag blind spots honestly.
 */
export async function fetchPullRequest(ref: PrRef) {
    const diffResponse = await fetch(
        `https://github.com/${ref.owner}/${ref.repo}/pull/${ref.number}.diff`,
        {
            headers: { "User-Agent": "pollinations-pr-review-fixer" },
        },
    );
    if (!diffResponse.ok) {
        throw new Error(
            `GitHub diff for PR #${ref.number} failed: ${diffResponse.status}`,
        );
    }
    const fullDiff = await diffResponse.text();

    let meta: {
        html_url?: string;
        title?: string;
        state?: string;
        draft?: boolean;
        user?: { login?: string };
        head?: { ref?: string; sha?: string };
    } = {};
    try {
        const apiResponse = await github(
            `/repos/${ref.owner}/${ref.repo}/pulls/${ref.number}`,
            "application/vnd.github+json",
        );
        meta = (await apiResponse.json()) as typeof meta;
    } catch {
        // API throttled: recover title and head branch from the PR page.
        try {
            const page = await fetch(
                `https://github.com/${ref.owner}/${ref.repo}/pull/${ref.number}`,
                { headers: { "User-Agent": "pollinations-pr-review-fixer" } },
            );
            if (page.ok) {
                const html = await page.text();
                const headRef = parseHeadRef(html);
                meta = {
                    html_url: `https://github.com/${ref.owner}/${ref.repo}/pull/${ref.number}`,
                    head: headRef ? { ref: headRef } : undefined,
                };
            }
        } catch {
            /* head ref stays unresolved; commands note it below */
        }
    }
    return { meta, diff: capDiff(fullDiff) };
}

/**
 * The exact operator commands: check out the PR head, apply the patch,
 * commit, and push WITHOUT force — only the reviewed PR's head branch is
 * ever written. Exported for tests.
 */
export function fixCommands(
    ref: PrRef,
    headRef: string,
    message: string,
): string[] {
    const subject = message.replace(/[\r\n]+/g, " ").slice(0, 72);
    return [
        `git clone https://github.com/${ref.owner}/${ref.repo}.git && cd ${ref.repo}`,
        `git fetch origin pull/${ref.number}/head:pr-${ref.number} && git checkout pr-${ref.number}`,
        "git apply fix.patch",
        `git commit -am "${subject.replaceAll('"', '\\"')}"`,
        `git push origin HEAD:${headRef}`,
    ];
}

const SYSTEM = `You are a meticulous pull-request reviewer. Review the given unified diff of a GitHub pull request.

Rules:
- Look for real defects only: bugs, security holes, correctness regressions, broken contracts. Style opinions are not findings.
- Report findings ordered by severity (critical, major, minor), each with file, title and a concrete detail.
- Then fix exactly ONE finding — the most important fixable one — with a minimal unified diff that applies cleanly with \`git apply\` against the PR head shown in the diff. Keep hunk headers accurate; do not touch unrelated lines.
- The patch must not add dependencies, secrets, network calls to new hosts, or merge/force-push behaviour.
- Answer with ONE fenced json block and nothing else, shaped exactly as:
{"summary":"string","findings":[{"severity":"critical|major|minor","file":"path","title":"string","detail":"string"}],"fix":{"finding":"title of the finding being fixed","patch":"unified diff text","verify":"shell command that proves the fix, e.g. a focused test"}}`;

/**
 * Pull the assistant's text out of a Responses (or Chat Completions) JSON
 * payload leniently. Exported for tests.
 */
export function outputTextOf(payload: unknown): string {
    if (!payload || typeof payload !== "object") return "";
    const record = payload as { output?: unknown; choices?: unknown };
    if (Array.isArray(record.output)) {
        return record.output
            .map((item) => {
                const content = (item as { content?: unknown })?.content;
                return Array.isArray(content)
                    ? content
                          .map(
                              (part) =>
                                  (part as { text?: unknown })?.text ?? "",
                          )
                          .join("")
                    : "";
            })
            .join("");
    }
    if (Array.isArray(record.choices)) {
        return record.choices
            .map(
                (choice) =>
                    (choice as { message?: { content?: unknown } })?.message
                        ?.content ?? "",
            )
            .join("");
    }
    return "";
}

/** One review call; falls back to the cheap model when the preferred one fails. */
export async function review(
    respond: AgentContext["respond"],
    model: (id: string) => unknown,
    diff: string,
    truncated: boolean,
): Promise<{ report: ReviewReport | null; modelUsed: string; raw: string }> {
    const prompt = `${truncated ? "NOTE: the diff was truncated before review.\n" : ""}Review this pull request diff:\n\n${diff}`;
    let modelUsed = PREFERRED_MODEL;
    let response = await respond({
        model: model(PREFERRED_MODEL),
        instructions: SYSTEM,
        prompt,
    });
    if (!response.ok) {
        modelUsed = FALLBACK_MODEL;
        response = await respond({
            model: model(FALLBACK_MODEL),
            instructions: SYSTEM,
            prompt,
        });
    }
    let raw = "";
    try {
        raw = outputTextOf(await response.json());
    } catch {
        /* report stays null below */
    }
    return { report: extractReport(raw), modelUsed, raw };
}

/** What this agent deliberately refuses to do, returned with every answer. */
export const AUTOMATION_BOUNDARIES = [
    "Never merges a PR: merging stays a human or maintainer action.",
    "Never force-pushes; the emitted push command is a fast-forward write to the reviewed PR's head branch only.",
    "Holds no GitHub credentials: GitHub is read unauthenticated and the operator runs the push commands with their own credentials.",
    "Never writes outside the reviewed PR branch, and never edits review comments or other issues.",
];

export default async function agent({ request, model, respond }: AgentContext) {
    let text = "";
    try {
        text = inputTextOf(await request.json());
    } catch {
        /* empty body handled below */
    }

    const ref = parsePrRef(text);
    if (!ref) {
        return Response.json(
            {
                ok: false,
                error: "Include a pull request to review, e.g. `Review https://github.com/owner/repo/pull/123 and fix the top finding`.",
            },
            { status: 400 },
        );
    }

    try {
        const { meta, diff } = await fetchPullRequest(ref);
        const { report, modelUsed } = await review(
            respond,
            model,
            diff.text,
            diff.truncated,
        );
        const head = meta.head;
        // Unknown head refs use a placeholder the operator must replace:
        // never guess the base branch, that would push to the wrong ref.
        const headRef = head?.ref ?? "PR_HEAD_BRANCH";
        const fixMessage = report?.fix?.finding
            ? `fix: ${report.fix.finding}`
            : `fix: review finding in #${ref.number}`;
        return Response.json({
            ok: true,
            model: modelUsed,
            pr: {
                url:
                    meta.html_url ??
                    `https://github.com/${ref.owner}/${ref.repo}/pull/${ref.number}`,
                title: meta.title,
                state: meta.state,
                draft: meta.draft,
                headRef,
                headSha: head?.sha,
                author: meta.user?.login,
            },
            review: report
                ? {
                      summary: report.summary,
                      findings: report.findings,
                      fix: {
                          finding: report.fix.finding,
                          verify: report.fix.verify,
                          patchFile: "fix.patch",
                          patch: report.fix.patch,
                          commands: fixCommands(ref, headRef, fixMessage),
                      },
                  }
                : null,
            diffTruncated: diff.truncated,
            boundaries: AUTOMATION_BOUNDARIES,
        });
    } catch (error) {
        return Response.json(
            {
                ok: false,
                error: error instanceof Error ? error.message : String(error),
            },
            { status: 502 },
        );
    }
}
