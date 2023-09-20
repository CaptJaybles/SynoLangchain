from typing import List, Union
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        elif "AI:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("AI:")[-1].strip()},
                log=llm_output,
            )
        elif "Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Answer:")[-1].strip()},
                log=llm_output,
            )
        elif "Assistant:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Assistant:")[-1].strip()},
                log=llm_output,
            )
        elif "Observation:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Observation:")[-1].strip()},
                log=llm_output,
            )
        elif "Output:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Output:")[-1].strip()},
                log=llm_output,
            )
        elif "output:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("output:")[-1].strip()},
                log=llm_output,
            )
        else:
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)