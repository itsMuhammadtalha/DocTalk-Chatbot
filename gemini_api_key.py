import google.generativeai as genai

genai.configure(api_key="AIzaSyCAo41zFdx6v__JvIRZIPo18Zvpa60c_Dg")

# Initialize the Gemini Pro model
model = genai.GenerativeModel("gemini-pro")

# Generate a response
response = model.generate_content("What is AI?")
print(response.text)