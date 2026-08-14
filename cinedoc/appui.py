from dotenv import load_dotenv
import streamlit as st
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI


load_dotenv()

model = ChatMistralAI(
    model="mistral-small-2603"
)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an intelligent movie information extraction assistant.

Your task is to analyze a paragraph containing information about a movie
and extract all useful and explicitly available movie-related information.

You must:
- Extract information only from the provided text.
- Never invent, assume, or hallucinate information.
- If a particular field is not available, return "Not mentioned".
- Identify multiple cast members when they are mentioned.
- Keep extracted information concise and accurate.
- Generate a short 2–4 sentence summary based only on the provided text.

Extract the following information:
1. Movie Name
2. Director
3. Cast
4. Genre
5. Release Year
6. Language
7. Production House
8. Country
9. Runtime
10. Rating
11. Plot / Story
12. Awards or Achievements
13. Other Important Details

Return the result in this exact format:

Movie Name:
Director:
Cast:
Genre:
Release Year:
Language:
Production House:
Country:
Runtime:
Rating:
Plot / Story:
Awards or Achievements:
Other Important Details:

Quick Summary:
"""
    ),
    (
        "human",
        """
Analyze the following movie-related paragraph:

{paragraph}
"""
    )
])


st.title("🎬 Movie Information Extractor")


# Larger input box
paragraph = st.text_area(
    "Enter your movie paragraph:",
    height=250,
    placeholder="Enter the movie details here..."
)


if st.button("Extract Movie Information"):

    if paragraph.strip():

        final_prompt = prompt.invoke({
            "paragraph": paragraph
        })

        response = model.invoke(final_prompt)

        result = response.content.strip()

        # Separate summary
        if "Quick Summary:" in result:
            movie_info, summary = result.split(
                "Quick Summary:",
                1
            )
        else:
            movie_info = result
            summary = ""

        st.subheader("Movie Information")

        fields = [
            "Movie Name",
            "Director",
            "Cast",
            "Genre",
            "Release Year",
            "Language",
            "Production House",
            "Country",
            "Runtime",
            "Rating",
            "Plot / Story",
            "Awards or Achievements",
            "Other Important Details"
        ]

        # Extract each field regardless of whether
        # the LLM put them on separate lines
        for i, field in enumerate(fields):

            if i < len(fields) - 1:
                next_field = fields[i + 1]

                pattern = (
                    rf"{re.escape(field)}:\s*(.*?)"
                    rf"(?=\s*{re.escape(next_field)}:)"
                )
            else:
                pattern = (
                    rf"{re.escape(field)}:\s*(.*)"
                )

            match = re.search(
                pattern,
                movie_info,
                re.IGNORECASE | re.DOTALL
            )

            if match:
                value = match.group(1).strip()

                st.markdown(
                    f"**{field}:** {value}"
                )

        # Summary
        if summary.strip():

            st.subheader("Quick Summary")

            st.write(summary.strip())

    else:

        st.warning("Please enter a movie paragraph.")