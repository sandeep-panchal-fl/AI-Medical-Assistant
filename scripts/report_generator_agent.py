from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import yaml
class ReportGeneratorAgent:

    def __init__(self):
        
        self.model = "gemma3:latest"
        self.base_url = "http://localhost:11434"
        
        # Create the Ollama model
        self.llm_chat = ChatOllama(
            model=self.model, 
            base_url=self.base_url,
        ) 

        # Load prompts
        with open('prompts.yaml', 'r') as f:
            prompts = yaml.safe_load(f)

        self.report_generator_prompt = prompts['medical_assistant']['report_generator_prompt']


    def generate_final_medical_report(self, full_chat, chat_summary, retrieved_knowledge=None):

        print("____________________________________\n")
        print("=== Generating Final Report ===")
        # severity_flag = self.calculate_severity_flag(chat_summary, full_chat)
        severity_flag = None
        
        enhanced_prompt = f"""
        {self.report_generator_prompt}
        
        ADDITIONAL CONTEXT:
        - Calculated Severity Flag: {severity_flag}
        """

        # - Current Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        
        report = self.llm_chat.invoke([
            SystemMessage(content=enhanced_prompt),
            HumanMessage(content=f"""
            CLINICAL SUMMARY: {chat_summary}
            
            RETRIEVED MEDICAL KNOWLEDGE: {retrieved_knowledge}
            
            FULL CONVERSATION: {full_chat}
            """)
        ])
        
        print("=== Final Report Generated ===")

        return report.content
    
if __name__=="__main__":
    pass

    chat_summary = """Patient reports a throbbing headache with sensitivity to light and nausea,
            onset not precisely determined. The headache is characterized as 10/10 in severity and is located
            diffusely throughout the head. Patient reports nausea alongside the headache, with the onset coinciding
            with the headache's initiation. Clinical examination reveals associated symptoms of photophobia and
            significant discomfort. Further evaluation of the patient’s cardiovascular and pulmonary systems is
            warranted given the acute nature of the symptoms.
        """
            
    full_chat = [
        {'HumanMessage': 'hi i am sandeep'},
        {'AIMessage': 'Hello Sandeep! I’m a medical assistant. What brings you in today?'},
        {'HumanMessage': 'i am having headache'},
        {'AIMessage': "I understand you're experiencing a headache, Sandeep. Can you tell me a little more about it – \
         where exactly do you feel the pain?"},
        {'HumanMessage': 'entire head'},
        {'AIMessage': 'I understand, Sandeep. That sounds tough. Can you describe the pain – \
         is it a throbbing, a dull ache, or something else?'},
        {'HumanMessage': 'throbbing'},
        {'AIMessage': 'Okay, a throbbing headache, Sandeep. When did this headache start, \
         and how severe would you rate it on a scale of 1 to 10, with 1 being very mild and 10 being the worst pain imaginable?'},
        {'HumanMessage': '10'},
        {'AIMessage': 'I understand, Sandeep. A 10 – that’s incredibly intense. \
         Are there any other symptoms you’re experiencing with the headache, like nausea, sensitivity to light,\
         or anything like that?'},
        {'HumanMessage': 'sensitivity to light and nausea'},
        {'AIMessage': 'Okay, Sandeep, so you’re experiencing a throbbing headache with sensitivity to \
         light and nausea. When did the nausea start alongside the headache?'},
        {'HumanMessage': 'stop'},
        {'AIMessage': 'STOP'}
        ]
    
    retrieved_knowledge = """Disease: Migraine | 
                Disease Description: A neurological condition characterized by intense, debilitating headaches. | 
                Precautions Recommendation: Avoid triggers like caffeine and stress, stay hydrated, get adequate sleep. | 
                Medicines Recommendation: ['Triptans', 'NSAIDs', 'Beta-blockers', 'Antidepressants'] |
                Diet Recommendation: ['Magnesium-rich foods', 'Avoid processed meats', 'Limit caffeine', 'Hydrating fluids'] |
                Workouts Recommendation: ['Yoga', 'Meditation', 'Stretching', 'Low-impact aerobic exercise']
                """
    
    obj = ReportGeneratorAgent()

    resp = obj.generate_final_medical_report(full_chat, chat_summary, retrieved_knowledge)
    print(resp)





#     ## Medical Assessment Report – Sandeep

# **Date:** October 26, 2023 (Generated Report)

# **1. PATIENT INFORMATION SECTION**

# *   **Patient Name:** Sandeep (Not specified)
# *   **Age:** Not specified
# *   **Gender:** Not specified

# **2. CLINICAL ASSESSMENT SECTION**

# *   **PRESENTING COMPLAINT:**
#     *   Primary symptom: Throbbing headache.
#     *   Duration: Not precisely determined, onset not precisely determined.
#     *   Onset: Acute, coinciding with the initiation of symptoms.
#     *   Severity: 10/10
# *   **SYMPTOM ANALYSIS:**
#     *   Location: Diffusely throughout the head.
#     *   Character: Throbbing.
#     *   Severity: 10/10 (HIGH) – Due to the reported intensity.
#     *   Aggravating/Relieving Factors: Not specified.
#     *   Associated Symptoms: Nausea, sensitivity to light.
# *   **SEVERITY FLAGGING LOGIC:**
#     *   HIGH: Severity â‰¥8
#     *   MEDIUM: Severity 5-7, multiple symptoms, functional impact
#     *   LOW: Severity â‰¤4, single mild symptom

# **3. MEDICAL RECOMMENDATIONS SECTION**

# *   **POTENTIAL CONSIDERATIONS:**
#     *   Differential diagnoses: Migraine, Tension Headache, Cluster Headache (Given the severity and associated symptoms, further investigation is warranted).
# *   **PRECAUTIONS & SELF-CARE:**
#     *   Immediate Do’s: Rest in a dark, quiet room.
#     *   Immediate Don’ts: Avoid strenuous activity.
#     *   Activity Modifications: Limit physical exertion.
#     *   Monitoring Advice: Monitor headache frequency, severity, and associated symptoms.
# *   **TREATMENT SUGGESTIONS:**
#     *   Medications: (Based on knowledge base - none currently available) Analgesics may be considered for symptomatic relief, but should be used with caution due to the severity.
#     *   Non-pharmacological approaches: Relaxation techniques, cold compresses.
#     *   Home remedies: Not specifically recommended at this stage.
# *   **MEDICAL FOLLOW-UP:**
#     *   When to seek urgent care: Seek immediate medical attention if headache worsens, develops new neurological symptoms (e.g., weakness, numbness, vision changes), or is unresponsive to initial treatment.
#     *   Recommended specialist if needed: Neurology consultation is recommended to investigate the etiology of the headache.
#     *   Timeline for re-evaluation: Within 24-48 hours, or sooner if symptoms worsen.

# **4. KNOWLEDGE & DISCLAIMER SECTION**

# These recommendations are based on generally accepted medical knowledge and the information provided. This report does not constitute a definitive diagnosis. Further evaluation and investigation by a qualified healthcare professional are essential.

# **Disclaimer:** This report is intended for informational purposes only and should not be considered a substitute for professional medical advice. The information presented is based on current medical knowledge and may be subject to change.

# ---

# **Note:** *This report is based solely on the provided clinical summary, retrieved knowledge (none), and conversation context. The lack of retrieved knowledge significantly limits the depth of the assessment and recommendations.*
