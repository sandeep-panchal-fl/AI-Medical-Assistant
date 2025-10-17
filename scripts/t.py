# Sidebar - STATIC PROGRESS TRACKING
with st.sidebar:
    st.header("🩺 Assistant Controls")
    st.markdown("""
    This AI medical assistant helps you by:
    - Asking about your symptoms  
    - Understanding their severity and duration  
    - Summarizing key clinical details  
    - Retrieving relevant medical insights  
    - Generating a concise medical summary report  

    **User Guidance:**
    - This tool is for informational purposes only and **not a substitute for professional medical advice**.  
    - Click ***‘End Conversation’*** button or send **‘stop’** in the chat to end the conversation anytime.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
    <div>💡 Click <strong>‘End Conversation’</strong> button or send <strong>‘stop’</strong> in the chat to end conversation and generate report</div>
    <div>⚠️ For emergencies, contact emergency services immediately</div>
    </div>
    """, 
    unsafe_allow_html=True
)