from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, EmailStr, field_validator
from dotenv import load_dotenv
import os


load_dotenv()


class CustomerSupportTicket(BaseModel):
    name: str = Field(description="Customer full name")
    email:str = Field(description="Customer email address")
    order_id: str = Field(description="Order ID")
    issue_type: str = Field(description="Type of issue")

    @field_validator("name")
    def validate_name(cls, value):
        if value.isdigit():
            raise ValueError("Name cannot contain only numbers")

        if not value.replace(" ", "").isalpha():
            raise ValueError("Name should contain only letters and spaces")

        return value



llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


structured_llm = llm.with_structured_output(CustomerSupportTicket)


text = """
Hi team,

My name is Sneha. I placed an order yesterday but the payment failed.

Order ID is ORD-45678.

My email is rahul.sharma@gmail.com.

Please help resolve this issue as soon as possible.

Thanks
Rahul
"""


prompt = f"""
You are an information extraction assistant.

Extract the following fields:

1. name
2. email
3. order_id
4. issue_type

If the customer mentions a payment failure,
set issue_type as "Payment Failed".

Customer Message:
{text}
"""

try:
    result = structured_llm.invoke(prompt)

    print("\nExtracted Data:")
    print(result)

    print("\nValidated JSON Output:")
    print(result.model_dump())

except Exception as e:
    print("\nValidation Error:")
    print(e)