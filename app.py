from time import perf_counter
import streamlit as st
from PIL import Image
from pathlib import Path
import json

from ai.classifier import (
    load_model,
    recognize_image,
    compare_cats_dogs,
    load_custom_cnn,
    predict_custom_cnn
)
from ai.flexible_recognition import (
    load_flexible_model,
    recognize_flexible,
    DEFAULT_CATEGORIES
)

# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="VisionAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# ==========================================
# 2. LOAD AI MODELS
# ==========================================

@st.cache_resource
def get_ai_model():
    return load_model()


@st.cache_resource
def get_custom_model():
    return load_custom_cnn()

@st.cache_resource
def get_flexible_model():
    return load_flexible_model()
# ==========================================
# 3. APPLICATION HEADER
# ==========================================

st.title("🧠 VisionAI")
st.subheader("Intelligent Image Recognition System")

st.sidebar.title("VisionAI")
st.sidebar.caption("CNN Image Classification Platform")


menu = st.sidebar.radio(
    "Choose an AI Module",
    [
        "Home",
        "Compare Two Categories",
        "Recognize an Image",
        "AI Model Laboratory",
        "Train Custom CNN",
        "Model Dashboard"
    ]
)


st.sidebar.divider()
st.sidebar.caption("Powered by TensorFlow & Streamlit")


# ==========================================
# 4. HOME PAGE
# ==========================================

if menu == "Home":

    st.header("Welcome to VisionAI")

    st.write(
        "An artificial intelligence application that "
        "uses convolutional neural networks to "
        "recognize and classify images."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🔍 AI Image Recognition")
        st.write(
            "Recognize objects using a pretrained "
            "MobileNetV2 convolutional neural network."
        )

    with col2:
        st.info("⚖️ CNN Model Comparison")
        st.write(
            "Compare predictions from MobileNetV2 "
            "and your own trained CNN."
        )

    with col3:
        st.info("🧠 Custom CNN Training")
        st.write(
            "Train a neural network using labeled "
            "cat and dog images."
        )

    st.divider()

    st.subheader("Available AI Models")

    st.write("**MobileNetV2:** Pretrained on ImageNet.")

    if (MODEL_DIR / "cats_vs_dogs.keras").exists():
        st.success("Custom Cats vs Dogs CNN: Available")
    else:
        st.warning("Custom Cats vs Dogs CNN: Not found")


# ==========================================
# 5. COMPARE TWO CATEGORIES
# ==========================================

elif menu == "Compare Two Categories":

    st.header("🐱 Cats vs Dogs 🐶")
    st.write(
        "Upload a picture and choose which CNN "
        "should analyze it."
    )

    model_choice = st.selectbox(
        "Choose AI Model",
        [
            "Pretrained MobileNetV2",
            "My Custom CNN"
        ]
    )

    uploaded_file = st.file_uploader(
        "Upload a cat or dog image",
        type=["jpg", "jpeg", "png"],
        key="comparison"
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

        except Exception as error:
            st.error(f"Unable to open image: {error}")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:

            st.subheader("CNN Analysis")

            if st.button(
                "🔍 Analyze with Selected Model",
                type="primary"
            ):

                try:
                    with st.spinner("Running CNN inference..."):

                        if model_choice == "My Custom CNN":

                            model = get_custom_model()

                            result = predict_custom_cnn(
                                image, model
                            )

                        else:

                            model = get_ai_model()

                            result = compare_cats_dogs(
                                image, model
                            )

                    st.session_state["comparison_result"] = result
                    st.session_state["comparison_model"] = model_choice
                    st.session_state["comparison_image"] = (
                        uploaded_file.getvalue()
                    )

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

            result = st.session_state.get("comparison_result")

            if (
                result is not None
                and st.session_state.get("comparison_model")
                == model_choice
                and st.session_state.get("comparison_image")
                == uploaded_file.getvalue()
            ):

                if result["prediction"] == "Uncertain":

                    st.warning(
                        "Insufficient evidence to identify "
                        "a cat or dog."
                    )

                else:

                    prediction = result["prediction"]

                    emoji = (
                        "🐱" if prediction == "Cat"
                        else "🐶"
                    )

                    st.success(
                        f"Prediction: {emoji} {prediction}"
                    )

                    st.write("### Classification Scores")

                    cat_score = result["cat"]
                    dog_score = result["dog"]

                    st.write(f"🐱 Cat: {cat_score:.2%}")
                    st.progress(cat_score)

                    st.write(f"🐶 Dog: {dog_score:.2%}")
                    st.progress(dog_score)

                    st.caption(
                        "Scores are model outputs or relative "
                        "class scores, not measured accuracy."
                    )

    st.divider()

    if model_choice == "My Custom CNN":
        st.info(
            "The custom CNN was trained from scratch "
            "using Cats vs Dogs images. It always "
            "predicts Cat or Dog, even for unrelated images."
        )
    else:
        st.info(
            "MobileNetV2 uses pretrained ImageNet categories "
            "and can return Uncertain when there is "
            "insufficient evidence."
        )


# ==========================================
# 6. GENERAL IMAGE RECOGNITION
# ==========================================


# ==========================================
# GENERAL IMAGE RECOGNITION
# ==========================================

elif menu == "Recognize an Image":

    st.header("🔍 VisionAI — Smart Image Recognition")

    st.write(
        "Recognize photographs, animals, cartoons, "
        "vehicles and other objects."
    )

    engine = st.radio(
        "Choose recognition engine",
        [
            "MobileNetV2 — ImageNet Recognition",
            "CLIP — Flexible Recognition"
        ],
        horizontal=True
    )

    if engine.startswith("CLIP"):

        st.info(
            "CLIP can compare images against "
            "user-defined descriptions, including "
            "cartoon animals and rabbits."
        )

        custom_categories = st.checkbox(
            "Customize recognition categories"
        )

        if custom_categories:
            category_text = st.text_area(
                "Enter one category per line",
                value="\n".join(DEFAULT_CATEGORIES),
                height=180
            )

            categories = [
                item.strip()
                for item in category_text.splitlines()
                if item.strip()
            ]
        else:
            categories = DEFAULT_CATEGORIES

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="smart_recognition"
    )

    if uploaded_file is not None:

        try:
            image = Image.open(
                uploaded_file
            ).convert("RGB")

        except Exception as error:
            st.error(f"Invalid image: {error}")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:

            st.subheader("AI Analysis")

            if st.button(
                "🧠 Recognize Image",
                type="primary"
            ):

                try:
                    with st.spinner(
                        "Analyzing image..."
                    ):

                        if engine.startswith("CLIP"):

                            model = get_flexible_model()

                            predictions = recognize_flexible(
                                image,
                                model,
                                categories
                            )

                        else:

                            model = get_ai_model()

                            original = recognize_image(
                                image,
                                model
                            )

                            predictions = [
                                {
                                    "category": item["category"],
                                    "score": item["confidence"]
                                }
                                for item in original
                            ]

                    st.session_state["smart_results"] = {
                        "file": uploaded_file.getvalue(),
                        "engine": engine,
                        "categories": (
                            tuple(categories)
                            if engine.startswith("CLIP")
                            else None
                        ),
                        "predictions": predictions
                    }

                except Exception as error:
                    st.error(
                        f"Recognition failed: {error}"
                    )

            saved = st.session_state.get(
                "smart_results"
            )

            current_categories = (
                tuple(categories)
                if engine.startswith("CLIP")
                else None
            )

            if (
                saved is not None
                and saved["file"] == uploaded_file.getvalue()
                and saved["engine"] == engine
                and saved["categories"] == current_categories
            ):

                st.success("Analysis complete!")

                for index, result in enumerate(
                    saved["predictions"], start=1
                ):

                    label = result["category"]
                    score = result["score"]

                    st.write(
                        f"**{index}. {label.title()}** "
                        f"— {score:.2%}"
                    )

                    st.progress(
                        min(1.0, max(0.0, score))
                    )

                if engine.startswith("CLIP"):

                    st.caption(
                        "CLIP scores describe relative "
                        "matches among the candidate "
                        "descriptions. They are not "
                        "calibrated probabilities."
                    )

                else:

                    st.caption(
                        "MobileNetV2 scores represent "
                        "ImageNet classification outputs."
                    )

# ==========================================
# 7. CUSTOM CNN TRAINING CENTER
# ==========================================

# ==========================================
# AI MODEL COMPARISON LABORATORY
# ==========================================

elif menu == "AI Model Laboratory":

    st.header("🔬 VisionAI Model Laboratory")

    st.write(
        "Compare the predictions of two different "
        "convolutional neural networks."
    )

    st.info(
        "Model A: Pretrained MobileNetV2 | "
        "Model B: Custom CNN trained from scratch"
    )

    uploaded_file = st.file_uploader(
        "Upload a cat or dog image",
        type=["jpg", "jpeg", "png"],
        key="laboratory_upload"
    )

    if uploaded_file is not None:

        try:
            image = Image.open(
                uploaded_file
            ).convert("RGB")

        except Exception as error:
            st.error(f"Invalid image: {error}")
            st.stop()

        st.image(
            image,
            caption="Test Image",
            width=350
        )

        image_bytes = uploaded_file.getvalue()

        if st.button(
            "🚀 Run Both CNN Models",
            type="primary"
        ):

            try:
                with st.spinner(
                    "Loading models and analyzing image..."
                ):

                    # MODEL A: MobileNetV2
                    pretrained = get_ai_model()

                    start_a = perf_counter()

                    result_a = compare_cats_dogs(
                        image,
                        pretrained
                    )

                    time_a = perf_counter() - start_a

                    # MODEL B: Custom CNN
                    custom = get_custom_model()

                    start_b = perf_counter()

                    result_b = predict_custom_cnn(
                        image,
                        custom
                    )

                    time_b = perf_counter() - start_b

                st.session_state["lab_results"] = {
                    "image": image_bytes,
                    "model_a": result_a,
                    "model_b": result_b,
                    "time_a": time_a,
                    "time_b": time_b
                }

            except Exception as error:
                st.error(f"Comparison failed: {error}")

        saved = st.session_state.get("lab_results")

        if saved and saved["image"] == image_bytes:

            result_a = saved["model_a"]
            result_b = saved["model_b"]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Model A")
                st.caption("Pretrained MobileNetV2")

                st.metric(
                    "Prediction",
                    result_a["prediction"]
                )

                if result_a["prediction"] != "Uncertain":
                    st.write(
                        f"🐱 Cat: {result_a['cat']:.2%}"
                    )
                    st.progress(result_a["cat"])

                    st.write(
                        f"🐶 Dog: {result_a['dog']:.2%}"
                    )
                    st.progress(result_a["dog"])

                else:
                    st.warning(
                        "Insufficient cat/dog evidence."
                    )

                st.metric(
                    "Inference Time",
                    f"{saved['time_a']:.3f} s"
                )

            with col2:
                st.subheader("Model B")
                st.caption("Custom CNN from scratch")

                st.metric(
                    "Prediction",
                    result_b["prediction"]
                )

                st.write(
                    f"🐱 Cat: {result_b['cat']:.2%}"
                )
                st.progress(result_b["cat"])

                st.write(
                    f"🐶 Dog: {result_b['dog']:.2%}"
                )
                st.progress(result_b["dog"])

                st.metric(
                    "Inference Time",
                    f"{saved['time_b']:.3f} s"
                )

            st.divider()
            st.subheader("Comparison Results")

            prediction_a = result_a["prediction"]
            prediction_b = result_b["prediction"]

            if prediction_a == "Uncertain":
                st.warning(
                    "MobileNetV2 rejected the image. "
                    "The custom CNN still made a "
                    "forced binary prediction."
                )

            elif prediction_a == prediction_b:
                st.success(
                    "Both CNN models agree: "
                    f"{prediction_a}"
                )

            else:
                st.warning(
                    "The CNN models disagree. "
                    f"MobileNetV2: {prediction_a}; "
                    f"Custom CNN: {prediction_b}"
                )

            st.caption(
                "Model A percentages are relative "
                "ImageNet category scores. Model B "
                "percentages are binary sigmoid outputs. "
                "They are not directly calibrated or "
                "equivalent confidence measurements. "
                "Inference times exclude model loading "
                "and may vary between runs."
            )

elif menu == "Train Custom CNN":

    st.header("🧠 Custom CNN Training Center")

    st.write(
        "Train a custom convolutional neural network "
        "to classify cats and dogs."
    )

    model_path = MODEL_DIR / "cats_vs_dogs.keras"

    if model_path.exists():

        st.success("A trained custom CNN is available!")

        st.write(
            f"**Saved model:** `{model_path.name}`"
        )

    else:

        st.warning("No trained custom CNN was found.")

    st.subheader("Training Configuration")

    st.number_input(
        "Training Epochs",
        min_value=1,
        max_value=100,
        value=20,
        disabled=True
    )

    st.number_input(
        "Batch Size",
        min_value=8,
        max_value=128,
        value=32,
        disabled=True
    )

    st.info(
        "Training is currently performed using "
        "ai/train_cnn.py in the VS Code terminal. "
        "Interactive training controls will be "
        "implemented in a later phase."
    )

    st.code(
        r".\.venv\Scripts\python.exe ai\train_cnn.py",
        language="powershell"
    )


# ==========================================
# 8. MODEL PERFORMANCE DASHBOARD
# ==========================================


elif menu == "Model Dashboard":

    st.header("📊 VisionAI — Model Performance Dashboard")

    comparison_path = MODEL_DIR / "model_comparison.json"
    confusion_path = MODEL_DIR / "confusion_matrices.png"
    training_path = MODEL_DIR / "training_results.png"

    if not comparison_path.exists():
        st.warning("Run ai/evaluate_models.py first.")

    else:
        with open(
            comparison_path,
            "r",
            encoding="utf-8"
        ) as file:
            results = json.load(file)

        mobile = results["mobilenet"]
        custom = results["custom_cnn"]

        st.subheader("Model Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### MobileNetV2")
            st.metric("Accuracy", f"{mobile['accuracy']:.2%}")
            st.metric("Macro F1", f"{mobile['macro_f1']:.4f}")
            st.metric(
                "Inference Time",
                f"{mobile['inference_seconds']:.2f} s"
            )
            st.metric(
                "Rejection Rate",
                f"{mobile['rejection_rate']:.2%}"
            )

        with col2:
            st.markdown("### Custom CNN")
            st.metric("Accuracy", f"{custom['accuracy']:.2%}")
            st.metric("Macro F1", f"{custom['macro_f1']:.4f}")
            st.metric(
                "Inference Time",
                f"{custom['inference_seconds']:.2f} s"
            )
            st.metric(
                "Rejection Rate",
                f"{custom['rejection_rate']:.2%}"
            )

        st.divider()

        st.subheader("Accuracy Comparison")

        import pandas as pd

        accuracy_df = pd.DataFrame({
            "Model": ["MobileNetV2", "Custom CNN"],
            "Accuracy (%)": [
                mobile["accuracy"] * 100,
                custom["accuracy"] * 100
            ]
        })

        st.bar_chart(
            accuracy_df.set_index("Model"),
            y="Accuracy (%)"
        )

        st.subheader("Precision, Recall and F1")

        rows = []

        for model_name, data in [
            ("MobileNetV2", mobile),
            ("Custom CNN", custom)
        ]:
            for i, category in enumerate(["Cat", "Dog"]):
                rows.append({
                    "Model": model_name,
                    "Category": category,
                    "Precision": data["precision"][i],
                    "Recall": data["recall"][i],
                    "F1": data["f1"][i],
                    "Support": data["support"][i]
                })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("Confusion Matrices")

        if confusion_path.exists():
            st.image(
                str(confusion_path),
                use_container_width=True
            )

        st.divider()

        st.subheader("Custom CNN Training History")

        if training_path.exists():
            st.image(
                str(training_path),
                use_container_width=True
            )

        st.info(
            f"Both models were evaluated on "
            f"{results['sample_count']} unseen images. "
            "Uncertain predictions count as incorrect "
            "for binary classification accuracy."
        )
