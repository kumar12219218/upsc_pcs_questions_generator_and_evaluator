import os
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

load_dotenv()

class MCQQuestion(BaseModel):
    question: str = Field(description="The question text")
    options: List[str] = Field(description="List of 4 possible answers")
    correct_answer: str = Field(description="The correct answer from the options")

    @validator('question', pre=True)
    def clean_question(cls, v):
        if isinstance(v, dict):
            return v.get('description', str(v))
        return str(v)
    
class FillBlankQuestion(BaseModel):
    question: str = Field(description="The question text with '______' for the blank")
    answer: str = Field(description="The correct word or phrase for the blank")

    @validator('question' , pre=True)
    def clean_question(cls, v):
        if isinstance(v, dict):
            return v.get('description', str(v))
        return str(v)


class QuestionGenerator:

    def __init__(self):
        """
        Initialize question with generator with Groq API
        """

        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0.9
        )
    
    def generate_mcq(self, topic:str, difficulty:str='medium')->MCQQuestion:
        """
        Generate multiple choice question with robust error handling
        """
        mcq_parser = PydanticOutputParser(pydantic_object=MCQQuestion)

        prompt=PromptTemplate(
            template=(
                "Generate a {difficulty} multiple-choice question about {topic}.\n\n"
                "Return ONLY a json object with these exact fields:\n"
                "-'question': A clear, specific question\n"
                "- 'options': An array of exactly 4 possible answers\n"
                "- 'correct_answer': One of the option that is the correct answer"
                "Example format:\n"
                '{{\n'
                '   "question": "What is the capital of France?",\n'
                '   "options": ["London","Berlin","Paris","Madrid"],\n'
                '   "correct_answer": "Paris"\n'
                '}}\n\n'
            ),
            input_variables=["topic", "difficulty"]
        )

        max_attempts=3
        for attempt in range(3):
            try:
                response=self.llm.invoke(prompt.format(topic=topic,difficulty=difficulty))
                parsed_response = mcq_parser.parse(response.content)

                if not parsed_response.question or len(parsed_response.options)!=4 or not       parsed_response.correct_answer:
                    raise ValueError("Invalid question format")
                if parsed_response.correct_answer not in parsed_response.options:
                    raise ValueError("Correct answer not in options")
                return parsed_response
            except Exception as e:
                if attempt == max_attempts-1:
                    raise RuntimeError(f"Failed to generate valid MCQ after {max_attempts} attempts: {str(e)}")
                continue
    
    def generate_fill_blank(self, topic: str, difficulty: str = 'medium') -> FillBlankQuestion:
        """
        Generate Fill in the Blank Question with robust error handling
        Includes:
        - Output parsing using Pydantic
        - Structured prompt template
        - Multiple retry attempts on failure
        - Validation of blank marker format
        """
        # Set up Pydantic parser for type checking and validation
        fill_blank_parser = PydanticOutputParser(pydantic_object=FillBlankQuestion)
        
        # Define the prompt template with specific format requirements
        prompt = PromptTemplate(
            template=(
                "Generate a {difficulty} fill-in-the-blank question about {topic}.\n\n"
                "Return ONLY a JSON object with these exact fields:\n"
                "- 'question': A sentence with '_____' marking where the blank should be\n"
                "- 'answer': The correct word or phrase that belongs in the blank\n\n"
                "Example format:\n"
                '{{\n'
                '    "question": "The capital of France is _____.",\n'
                '    "answer": "Paris"\n'
                '}}\n\n'
                "Your response:"
            ),
            input_variables=["topic", "difficulty"]
        )

        # Implement retry logic with maximum attempts
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Generate response using LLM
                response = self.llm.invoke(prompt.format(topic=topic, difficulty=difficulty))
                parsed_response = fill_blank_parser.parse(response.content)
                
                # Validate the generated question meets requirements
                if not parsed_response.question or not parsed_response.answer:
                    raise ValueError("Invalid question format")
                if "_____" not in parsed_response.question:
                    parsed_response.question = parsed_response.question.replace("___", "_____")
                    if "_____" not in parsed_response.question:
                        raise ValueError("Question missing blank marker '_____'")
                
                return parsed_response
            except Exception as e:
                # On final attempt, raise error; otherwise continue trying
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate valid fill-in-the-blank question after {max_attempts} attempts: {str(e)}")
                continue