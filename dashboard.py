import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image

# ---------- ESTILOS ----------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}

h1 {
    color: #2c3e50;
    text-align: center;
}

.stButton>button {
    background-color: #27ae60;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 20px;
}

.stButton>button:hover {
    background-color: #219150;
}

textarea {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Generador de Imágenes IA",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Generador de Imágenes Ultra Realistas")
st.markdown(
    "<p style='text-align:center; color:gray;'>Optimizado para cualquier objeto · Estilo fotográfico profesional</p>",
    unsafe_allow_html=True
)

# ---------- PROMPTS ----------
user_prompt = st.text_area(
    "Describe lo que quieres generar (cualquier cosa):",
    "Un perro pitbull negro con una raya blanca en la cara"
)

# Prompt realista universal (sirve para TODO)
realism_prompt = """
ultra realistic RAW photo, professional photography,
high detail, sharp focus, natural lighting,
cinematic light, realistic textures, depth of field,
85mm DSLR lens, high resolution, photorealistic
"""

# Prompt final combinado
final_prompt = f"{realism_prompt}, {user_prompt}"

# Negative prompt universal
negative_prompt = st.text_area(
    "Qué NO quieres en la imagen:",
    """
cartoon, anime, illustration, painting, CGI, 3D render,
low quality, blurry, deformed, bad anatomy,
extra limbs, extra fingers, unrealistic proportions,
plastic skin, oversaturated colors, watermark, logo, text
"""
)

seed = st.number_input("Seed (opcional)", value=12345)

# ---------- GENERACIÓN ----------
if st.button("🎨 Generar imagen"):
    with st.spinner("Generando imagen ultra realista..."):
        client = InferenceClient(
    model="stabilityai/sdxl-turbo",
    token=st.secrets["HUGGINGFACE_TOKEN"],
    provider="hf-inference"
)


        image = client.text_to_image(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
            guidance_scale=7.0,
            num_inference_steps=20,
            seed=seed
        )

        st.image(image, caption="Imagen generada", use_column_width=True)
        image.save("imagen_generada.png")
        st.success("Imagen guardada como imagen_generada.png")

