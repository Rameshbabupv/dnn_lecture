# Prompt: Build the Answer Key for Formative Test I

You are helping me prepare the answer sheet for the Formative Test I paper that was generated using the companion prompt. Follow these instructions exactly:

1. **Inputs**
   - The student-facing question paper is named `1_Question_paper.md` and is located in the same directory as this prompt.
   - MCQ answers are derived from `45-1mark-mcq.md` (same directory) by matching the question text.
   - SAQ model answers come from `7-5marks-saq.md` (same directory), matching the selected question prompts exactly.

2. **Part A (MCQs)**
   - For each MCQ appearing in `1_Question_paper.md`, identify the correct option from the MCQ source file and restate the question with its options.
   - Highlight only the correct option line by wrapping it in `<span style="color: #008000;">…</span>`.
   - After the options, include a line `**Answer**: <letter>) <text>` without any additional styling.
   - Immediately after the answer line, add a single-sentence justification showing why the chosen option is correct and why the alternatives do not apply, drawing phrasing from `45-1mark-mcq-explanations.md` wherever there is a matching explanation (otherwise rely on deep learning fundamentals).

3. **Part B (SAQs)**
   - For each SAQ appearing in `1_Question_paper.md`, copy the full model answer from the SAQ source file verbatim.
   - Retain the metadata line (module/week/difficulty/CO/source ID) exactly as in the question paper.

4. **Structure & Formatting**
   - Preserve the same section order as the question paper: title block, instructions summary (if any), Part A, Part B, distribution summary.
   - Keep text ASCII-only and adhere to existing markdown conventions (bold labels, horizontal rules, etc.).
   - Do not include any extra commentary beyond the answer content and required justification line.

5. **Consistency Checks**
   - Verify that each MCQ in the answer sheet matches an entry in the student paper; if a match is not found, state the issue and stop.
   - Ensure the distribution summary reflects the same counts as the question paper.

Produce only the markdown for the answer sheet; no explanations or additional notes.
