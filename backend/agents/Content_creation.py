from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
try:
    from backend.services.image_generation import generate_image
except ImportError:
    from services.image_generation import generate_image


# ================================
# 1. Output Schema (Pydantic)
# ================================
class ContentCreationOutput(BaseModel):
    caption: str = Field(description="Caption for the social media post")
    image_prompt: str = Field(description="Prompt used to generate the post image")
    hashtags: list[str] = Field(description="List of hashtags for the post")
    platform: str = Field(description="Platform (LinkedIn, Instagram, etc.)")


# ================================
# 2. LLM Setup (Bedrock)
# ================================
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)

# Parser
parser = PydanticOutputParser(pydantic_object=ContentCreationOutput)


# ================================
# 3. Agent Class
# ================================
class ContentCreationAgent:

    def research_tools(self, query: str) -> str:
        # TODO: Replace with Tavily or real search API
        prompt = f"Do research on: {query}"
        response = llm.invoke(prompt)
        return response.content

    def create_social_post(self, topic: str, platform: str, memory_context: dict | None = None) -> ContentCreationOutput:
        # Task 5.1: Build platform-specific prompt
        platform_lower = platform.lower()
        if platform_lower == "linkedin":
            prompt_text = (
                f"Create a professional LinkedIn post about {topic}. "
                "Use no more than 5 hashtags. "
                "Return JSON with caption, image_prompt, hashtags (list), platform."
            )
        else:  # instagram
            prompt_text = (
                f"Create a conversational Instagram post about {topic}. "
                "Use 10 to 30 hashtags. "
                "Return JSON with caption, image_prompt, hashtags (list), platform."
            )

        # Task 5.2: Memory context awareness — exclude recent hashtags
        if memory_context is not None:
            prior_posts = memory_context.get(f"{platform_lower}_posts", [])
            if prior_posts:
                recent_posts = prior_posts[-3:]
                excluded_hashtags = []
                for post in recent_posts:
                    excluded_hashtags.extend(post.get("hashtags", []))
                if excluded_hashtags:
                    prompt_text += f" Avoid these hashtags from recent posts: {excluded_hashtags}"

        full_prompt = f"{prompt_text}\n\n{parser.get_format_instructions()}"
        response = llm.invoke(full_prompt)
        output: ContentCreationOutput = parser.parse(response.content)

        # Task 5.3: Call generate_image (side effect; result not stored in ContentCreationOutput)
        try:
            generate_image(output.image_prompt)
        except Exception:
            pass

        return ContentCreationOutput(
            caption=output.caption,
            image_prompt=output.image_prompt,
            hashtags=output.hashtags,
            platform=platform,
        )


    def linkedin_content_creation(self, query: str) -> ContentCreationOutput:
        context = self.research_tools(query)

        prompt = f"""
        Create a professional LinkedIn post based on:

        {context}

        {parser.get_format_instructions()}
        """

        response = llm.invoke(prompt)

        parsed = parser.parse(response.content)
        return parsed


    def instagram_content_creation(self, query: str) -> ContentCreationOutput:
        context = self.research_tools(query)

        prompt = f"""
        Create an engaging Instagram caption with hashtags based on:

        {context}

        {parser.get_format_instructions()}
        """

        response = llm.invoke(prompt)

        parsed = parser.parse(response.content)
        return parsed


# ================================
# 4. Run Example
# ================================
# if __name__ == "__main__":
#     agent = ContentCreationAgent()

#     result = agent.linkedin_content_creation("AI in healthcare")

#     print("\nContent:\n", result.content)
#     print("\nPlatform:", result.platform)
#     print("\nResponse:", result.response)