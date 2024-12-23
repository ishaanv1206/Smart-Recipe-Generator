import streamlit as st
import cv2
import numpy as np
from PIL import Image
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from ultralytics import YOLO

@st.cache_resource
def load_yolov7():
    model = YOLO("runs/classify/train/weights/best.pt")
    return model

def detect_ingredients(image):
    model = load_yolov7()
    img = np.array(image)
    results = model(img)
    detected_items = set()
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            label = model.names[class_id]
            detected_items.add(label)
    return list(detected_items)

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key="", model_name="llama3-8b-8192")

    def extract_ingredients(self, image_description):
        prompt_extract = PromptTemplate.from_template(
            """
            ### IMAGE DESCRIPTION:
            {image_description}
            ### INSTRUCTION:
            The image description contains a list of detected kitchen ingredients. 
            Your job is to extract the ingredients and return them in JSON format containing the key: `ingredients`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"image_description": image_description})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse ingredients.")
        return res if isinstance(res, list) else [res]

    def generate_recipe(self, ingredients):
        prompt_recipe = PromptTemplate.from_template(
            """
            ### INGREDIENTS:
            {ingredients}
            
            ### INSTRUCTION:
            Generate a recipe using the following ingredients: {ingredients}. 
            Please make sure the recipe is simple and easy to follow. Add any extra details like cooking steps, serving sizes, etc.
            ### RECIPE (NO PREAMBLE):
            """
        )
        chain_recipe = prompt_recipe | self.llm
        res = chain_recipe.invoke({"ingredients": str(ingredients)})
        return res.content

def app():
    st.title('Smart Recipe Generator from Kitchen Images')
    st.write("Upload an image of your kitchen, and we'll detect ingredients and generate a recipe.")
    uploaded_image = st.file_uploader("Choose a kitchen image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Kitchen Image", use_column_width=True)
        with st.spinner('Detecting ingredients...'):
            ingredients = detect_ingredients(image)
        
        if ingredients:
            st.write(f"Detected ingredients: {', '.join(ingredients)}")
            with st.spinner('Generating recipe...'):
                chain = Chain()
                recipe = chain.generate_recipe(ingredients)
            st.subheader("Suggested Recipe:")
            st.write(recipe)
        else:
            st.write("No ingredients detected. Please upload a better image.")

if __name__ == "__main__":
    app()
