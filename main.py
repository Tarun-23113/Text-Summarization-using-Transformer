import streamlit as st
from transformers import pipeline

# Initialize the summarization pipeline with BART
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def main():
    st.title("📝 Text Summarization using BART")
    st.markdown("*Powered by Hugging Face Transformers - Encoder-Decoder Architecture*")
    
    # Sidebar for parameters
    st.sidebar.header("Summarization Settings")
    max_length = st.sidebar.slider("Max Summary Length", 50, 300, 130)
    min_length = st.sidebar.slider("Min Summary Length", 10, 100, 30)
    
    # Main text input
    text_input = st.text_area(
        "Enter text to summarize:",
        height=250,
        placeholder="Paste your article, document, or long text here..."
    )
    
    if st.button("Generate Summary", type="primary"):
        if text_input.strip():
            with st.spinner("Generating summary..."):
                try:
                    summarizer = load_summarizer()
                    summary = summarizer(
                        text_input,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    
                    st.success("Summary Generated!")
                    st.subheader("Summary:")
                    st.write(summary[0]['summary_text'])
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Length", f"{len(text_input.split())} words")
                    with col2:
                        st.metric("Summary Length", f"{len(summary[0]['summary_text'].split())} words")
                    with col3:
                        compression = (1 - len(summary[0]['summary_text'].split()) / len(text_input.split())) * 100
                        st.metric("Compression", f"{compression:.1f}%")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter some text to summarize.")
    
    # Example text
    with st.expander("📌 Try an example"):
        example_text = """The transformer architecture has revolutionized natural language processing since its introduction in 2017. 
        Unlike recurrent neural networks, transformers process entire sequences simultaneously using self-attention mechanisms. 
        This parallel processing capability makes them significantly faster to train on modern hardware. The encoder-decoder 
        structure is particularly effective for sequence-to-sequence tasks like translation and summarization. The encoder 
        processes the input text and creates contextual representations, while the decoder generates the output sequence. 
        Attention mechanisms allow the model to focus on relevant parts of the input when generating each output token. 
        Pre-trained models like BART, T5, and GPT have achieved state-of-the-art results across numerous NLP benchmarks. 
        These models are trained on massive text corpora and can be fine-tuned for specific downstream tasks with relatively 
        small datasets. The success of transformers has led to their adoption beyond NLP, including computer vision and 
        multimodal applications."""
        
        if st.button("Load Example"):
            st.rerun()

if __name__ == "__main__":
    main()
