GRADER_TEMPLATE = """
Your job is to evaluate an agent's response to a code comprehension question against a specific rubric criterion.

# Task Prompt
The agent was asked the following question about a codebase:

<<task_prompt>>

# Reference Answer
An expert provided this reference answer for context:

<<reference_answer>>

# Agent Response
The agent provided this response:

<<agent_response>>

# Rubric Criterion
<<rubric_criterion>>

# Criterion Type
<<criterion_type>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the agent's response does or does not meet the criterion.
- The "criteria_met" field should be a boolean indicating whether the response meets the criterion.
- If a criterion has multiple sentences or sub-criteria, consider all of them. If any sub-criterion is not met, return false. Only return true if all parts are met.
- One important exception: if a criterion says "such as", "for example", or "including", the response does not have to include all listed examples to meet the criterion.

# Understanding Criterion Types

## For "positive hli verifier" criteria:
These describe factual claims or behaviors that the response SHOULD contain.
- Return criteria_met: true if the agent's response correctly addresses and includes this information.
- Return criteria_met: false if the agent's response fails to include or incorrectly states this information.

## For "negative hli verifier" criteria:
These describe undesirable behaviors or incorrect claims that the response should NOT exhibit.
- Return criteria_met: true if the agent's response DOES exhibit this undesirable behavior (this is bad).
- Return criteria_met: false if the agent's response does NOT exhibit this undesirable behavior (this is good).

In other words, for negative criteria, a good response should result in criteria_met: false because it avoids the undesirable behavior.

# Examples

Example 1 (positive verifier):
Criterion: "Identifies that the server uses Unix domain sockets for IPC"
Agent response mentions Unix domain sockets → criteria_met: true
Agent response does not mention sockets → criteria_met: false

Example 2 (negative verifier):
Criterion: "Claims the server uses TCP sockets when it actually uses Unix domain sockets"
Agent response correctly says Unix domain sockets → criteria_met: false (good - did not make the wrong claim)
Agent response incorrectly says TCP sockets → criteria_met: true (bad - made the wrong claim)

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()
