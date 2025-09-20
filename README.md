## 💡 Project: **"VIP Complaint Prediction & Root Cause AI"**

### 📌 Background
VIP & Enterprise customers are **high value** — large contracts, significant revenue contribution, and operator reputation at stake.  
Currently, complaint handling is **reactive** → team only acts after a complaint is received.  
Problem: **delayed response**. Impact: contracts may incur penalties, customer trust decreases.

🔑 Solution: **AI Early Warning System** → predict potential complaints **before** customers raise them.

### 🎯 Objectives
1. **Complaint Prediction** → detect sites/clusters at risk of VIP complaints within the next 24 hours.  
2. **Root Cause Identification** → automate diagnosis based on KPI patterns (e.g., high PRB → slow internet).  
3. **Prioritization Alert** → send early warning notifications to the Network Engineer team.

### 🔑 Inputs (X – Features)
* **Radio KPIs**: RSRP, RSRQ, SINR, PRB DL/UL, Max/Active users, UL interference, HO success rate, CSSR.  
* **Core & Transport KPIs**: Packet loss, latency, CPRI/VSWR alarms, TNL availability.  
* **Service-Specific KPIs**: WhatsApp success ratio, VoLTE drop rate, SMS success ratio, Gaming latency.  
* **Events/External**: Maintenance schedules, concerts, national events, weather conditions.

### 🎯 Outputs (y – Target)
**Level 1 – Complaint Prediction (Binary)**  
Will the site/cluster likely generate a complaint in the next 24 hours? (Yes / No)

**Level 2 – Complaint Category (Multiclass)**  
Predicted complaint types:
- Internet Lag  
- Coverage / Signal Issues  
- WhatsApp Call Issues  
- Voice Call Issues  
- SMS Issues  
- Gaming Lag

**Level 3 – Root Cause Recommendation**  
Map prediction results to potential root causes:
- High PRB → capacity overload  
- Low RSRP → weak coverage  
- High UL interference → poor quality  
- Transport packet loss → core/transport problem

### 🚀 Value Proposition
- ✅ Shift from **Reactive → Proactive** VIP complaint handling.  
- ✅ **Improve SLA & customer experience** → prevent complaints before escalation.  
- ✅ Scalable to **all customer complaints**, not just VIP.  

### 🔗 Try the Demo
<a href="https://vip-complaint-prediction.streamlit.app/" target="_blank">Streamlit App</a>
