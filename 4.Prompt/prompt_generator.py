from langchain_core.prompts import PromptTemplate


template = PromptTemplate(
    input_variables=["paper", "style", "length"],
    validate_template=True,
    template="Summarize the research paper on '{paper}' in a '{style}' style within {length} words.",
)

template.save("research_summary_template.json")