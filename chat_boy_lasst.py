import pandas as pd
import gradio as gr
from PIL import Image
import os
from dotenv import load_dotenv
from datetime import datetime
import hashlib
from openai import OpenAI
import PyPDF2
import qrcode
import uuid

def home():
    return "Welcome to Athletes ChatBoy! 🧠\n\nUse the tabs above to navigate."

# Load environment variables
load_dotenv(dotenv_path=r"C:\Users\Grace\Downloads\New folder\New Scipt using Gradio\.env")

csv_file = 'athletes_data.csv'
archive_file = 'archived_seniors.csv'
coaches_file = 'coach_accounts.csv'
activity_log = 'coach_activity_log.csv'
schedule_file = 'schedules.csv'
equipment_file = 'equipment_purchases.csv'
supplies_file = 'trainer_supplies.csv'
pay_file = 'coach_pay.csv'
phone_file = 'phone_directory.csv'
qr_revenue_file = 'qr_revenue.csv'
BUDGET = 47000.00  # Annual budget

# --- PDF Manual Extraction ---
def extract_pdf_text(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        text = ""
    return text

# Paths to your manuals
FOOTBALL_MANUAL_PATH = "2425_football_sport_manual_update_12_3_24.pdf"
BASKETBALL_MANUAL_PATH = "2324_basketball_sport_manual.pdf"
VOLLEYBALL_MANUAL_PATH = "2425_volleyball_sport_manual.pdf"
SOFTBALL_MANUAL_PATH = "2425_softball_sport_manual_update (1).pdf"
BASEBALL_MANUAL_PATH = "2425_baseball_sport_manual.pdf"

# Preload the manual texts for searching
football_manual_text = extract_pdf_text(FOOTBALL_MANUAL_PATH) if os.path.exists(FOOTBALL_MANUAL_PATH) else ""
basketball_manual_text = extract_pdf_text(BASKETBALL_MANUAL_PATH) if os.path.exists(BASKETBALL_MANUAL_PATH) else ""
volleyball_manual_text = extract_pdf_text(VOLLEYBALL_MANUAL_PATH) if os.path.exists(VOLLEYBALL_MANUAL_PATH) else ""
softball_manual_text = extract_pdf_text(SOFTBALL_MANUAL_PATH) if os.path.exists(SOFTBALL_MANUAL_PATH) else ""
baseball_manual_text = extract_pdf_text(BASEBALL_MANUAL_PATH) if os.path.exists(BASEBALL_MANUAL_PATH) else ""

def load_sports_list():
    if os.path.exists("sports_list.csv"):
        return pd.read_csv("sports_list.csv")["Sport"].tolist()
    return [
        "Football", "Flag Football", "Volleyball", "Basketball", "Soccer", "Wrestling",
        "Cross Country", "Track and Field", "Baseball", "Softball"
    ]

def load_manual_map():
    if os.path.exists("manual_map.csv"):
        df = pd.read_csv("manual_map.csv")
        return dict(zip(df["Sport"], df["ManualFile"]))
    return {
        "Football": FOOTBALL_MANUAL_PATH,
        "Basketball": BASKETBALL_MANUAL_PATH,
        "Baseball": BASEBALL_MANUAL_PATH,
        "Softball": SOFTBALL_MANUAL_PATH,
        "Volleyball": VOLLEYBALL_MANUAL_PATH
    }

def load_roles_list():
    if os.path.exists("roles_list.csv"):
        return pd.read_csv("roles_list.csv")["Role"].tolist()
    return ["Referee", "Trainer", "Security", "Other Staff"]

sports_list = load_sports_list()
manual_map = load_manual_map()
roles_list = load_roles_list()

GIRLS_SPORTS = [
    "Volleyball", "Softball", "Girls Basketball", "Girls Soccer",
    "Girls Cross Country", "Girls Track and Field"
]
GIRLS_BUDGET = 5000.00

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- AI Chatbot with OpenAI (new API) ---
def ask_chatboy(question):
    # Gather context from all manuals
    manuals = {
        "football": football_manual_text,
        "basketball": basketball_manual_text,
        "volleyball": volleyball_manual_text,
        "softball": softball_manual_text,
        "baseball": baseball_manual_text
    }
    context = ""
    for sport, text in manuals.items():
        if sport in question.lower() and text:
            context = text
            break
    if not context:
        context = "\n".join(manuals.values())
    context = context[:3000]
    prompt = (
        f"You are an expert on high school sports rules. "
        f"Here is some manual text:\n{context}\n\n"
        f"Answer this question in plain language: {question}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI error: {e}"

def submit_athlete(
    name, dob, age, school, sport, position, physical_date, physical_expiry,
    grad_year, grade9_entry, impact_test, home_phone, cell_phone, email, gpa,
    football_ins, interscholastic_ins, birth_cert_file, parent_dl_file
):
    if not name or not dob or not school or not sport:
        return "Please fill in all required fields (Name, DOB, School, Sport)."
    try:
        dob = str(dob)
        physical_date = str(physical_date)
        physical_expiry = str(physical_expiry)
        grade9_entry = str(grade9_entry)
        impact_test = str(impact_test)
    except Exception:
        return "Invalid date format."

    data = {
        "Name": name,
        "Date of Birth": dob,
        "Age": age,
        "School": school,
        "Sport": sport,
        "Position": position,
        "Physical Date": physical_date,
        "Physical Expiry Date": physical_expiry,
        "Graduation Year": grad_year,
        "9th Grade Entry Date": grade9_entry,
        "Baseline IMPACT Test Date": impact_test,
        "Home Phone": home_phone,
        "Cell Phone": cell_phone,
        "Email": email,
        "GPA": gpa,
        "Football Insurance Status": football_ins,
        "Interscholastic Insurance Status": interscholastic_ins
    }
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_new = pd.DataFrame([data])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_file, index=False)
    else:
        df = pd.DataFrame([data])
        df.to_csv(csv_file, index=False)

    os.makedirs("uploads/birth_certificates", exist_ok=True)
    os.makedirs("uploads/parent_ids", exist_ok=True)

    if birth_cert_file is not None:
        birth_cert_path = f"uploads/birth_certificates/{name.replace(' ', '_')}_birth_cert.{birth_cert_file.name.split('.')[-1]}"
        with open(birth_cert_path, "wb") as f:
            f.write(birth_cert_file.read())

    if parent_dl_file is not None:
        parent_dl_path = f"uploads/parent_ids/{name.replace(' ', '_')}_parent_id.{parent_dl_file.name.split('.')[-1]}"
        with open(parent_dl_path, "wb") as f:
            f.write(parent_dl_file.read())

    return "Athlete information saved!"

def create_coach_account(new_name, new_pw, confirm_pw, cert_file, cert_expiry, sport):
    if not new_name or not new_pw or not confirm_pw or not cert_expiry or not sport:
        return "Please fill in all required fields."
    if new_pw != confirm_pw:
        return "Passwords do not match."
    hashed_pw = hashlib.sha256(new_pw.encode()).hexdigest()
    coach_data = pd.DataFrame([[new_name, hashed_pw, cert_expiry, sport]], columns=['Coach Name', 'Password', 'Cert Expiry', 'Sport'])
    if os.path.exists(coaches_file):
        existing = pd.read_csv(coaches_file)
        if new_name in existing['Coach Name'].values:
            return "Coach with this name already exists."
        else:
            updated = pd.concat([existing, coach_data], ignore_index=True)
            updated.to_csv(coaches_file, index=False)
    else:
        coach_data.to_csv(coaches_file, index=False)

    if cert_file is not None:
        os.makedirs("uploads/coach_certifications", exist_ok=True)
        cert_path = f"uploads/coach_certifications/{new_name.replace(' ', '_')}_cert.{cert_file.name.split('.')[-1]}"
        with open(cert_path, "wb") as f:
            f.write(cert_file.read())
    return "Account created."

def view_archived_athletes(sport_filter=None):
    if os.path.exists(archive_file):
        df = pd.read_csv(archive_file)
        if sport_filter and sport_filter != "All":
            df = df[df["Sport"] == sport_filter]
        return df
    else:
        return pd.DataFrame(columns=["No archived athletes found."])

def add_schedule(date, opponent, location, bus_cost, referee_pay, trainer_pay, security_pay, other_pay, booking_fee):
    if not date or not opponent or not location:
        return "Please fill in all required fields (Date, Opponent, Location)."
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except Exception:
        return "Invalid date format. Use YYYY-MM-DD."
    data = {
        "Date": date,
        "Opponent": opponent,
        "Location": location,
        "Bus Cost": float(bus_cost) if location == "Away" else 0.0,
        "Referee Pay": float(referee_pay) if location == "Home" else 0.0,
        "Trainer Pay": float(trainer_pay) if location == "Home" else 0.0,
        "Security Pay": float(security_pay) if location == "Home" else 0.0,
        "Other Staff Pay": float(other_pay) if location == "Home" else 0.0,
        "Booking Fee": float(booking_fee) if location == "Home" else 0.0
    }
    if os.path.exists(schedule_file):
        df = pd.read_csv(schedule_file)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    else:
        df = pd.DataFrame([data])
    df.to_csv(schedule_file, index=False)
    return "Schedule added!"

def view_schedules():
    if os.path.exists(schedule_file):
        return pd.read_csv(schedule_file)
    else:
        return pd.DataFrame(columns=[
            "Date", "Opponent", "Location", "Bus Cost",
            "Referee Pay", "Trainer Pay", "Security Pay", "Other Staff Pay", "Booking Fee"
        ])

def schedule_summary():
    if not os.path.exists(schedule_file):
        return pd.DataFrame(columns=["No schedules found."])
    df = pd.read_csv(schedule_file)
    total_bus = df.get("Bus Cost", pd.Series()).sum()
    total_ref = df.get("Referee Pay", pd.Series()).sum()
    total_trainer = df.get("Trainer Pay", pd.Series()).sum()
    total_security = df.get("Security Pay", pd.Series()).sum()
    total_other = df.get("Other Staff Pay", pd.Series()).sum()
    total_booking = df.get("Booking Fee", pd.Series()).sum()
    summary = pd.DataFrame({
        "Total Bus Cost": [total_bus],
        "Total Referee Pay": [total_ref],
        "Total Trainer Pay": [total_trainer],
        "Total Security Pay": [total_security],
        "Total Other Staff Pay": [total_other],
        "Total Booking Fees": [total_booking],
        "Total Officials/Staff Cost": [total_ref + total_trainer + total_security + total_other + total_booking]
    })
    return summary

def add_equipment_purchase(sport, item, amount):
    data = {
        "Sport": sport,
        "Item": item,
        "Amount": float(amount),
        "Date": datetime.now().strftime("%Y-%m-%d")
    }
    if os.path.exists(equipment_file):
        df = pd.read_csv(equipment_file)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    else:
        df = pd.DataFrame([data])
    df.to_csv(equipment_file, index=False)
    return "Equipment purchase logged!"

def add_trainer_supply(item, amount):
    data = {
        "Item": item,
        "Amount": float(amount),
        "Date": datetime.now().strftime("%Y-%m-%d")
    }
    if os.path.exists(supplies_file):
        df = pd.read_csv(supplies_file)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    else:
        df = pd.DataFrame([data])
    df.to_csv(supplies_file, index=False)
    return "Trainer supply purchase logged!"

def budget_status():
    spent = 0.0
    # Schedules
    if os.path.exists(schedule_file):
        df = pd.read_csv(schedule_file)
        spent += df.get("Bus Cost", pd.Series()).sum()
        spent += df.get("Referee Pay", pd.Series()).sum()
        spent += df.get("Trainer Pay", pd.Series()).sum()
        spent += df.get("Security Pay", pd.Series()).sum()
        spent += df.get("Other Staff Pay", pd.Series()).sum()
        spent += df.get("Booking Fee", pd.Series()).sum()
    # Equipment
    if os.path.exists(equipment_file):
        df = pd.read_csv(equipment_file)
        spent += df.get("Amount", pd.Series()).sum()
    # Trainer supplies
    if os.path.exists(supplies_file):
        df = pd.read_csv(supplies_file)
        spent += df.get("Amount", pd.Series()).sum()
    remaining = BUDGET - spent
    summary = pd.DataFrame({
        "Annual Budget": [BUDGET],
        "Total Spent": [spent],
        "Remaining Budget": [remaining]
    })
    return summary

def girls_budget_status():
    spent = 0.0
    # Equipment
    if os.path.exists(equipment_file):
        df = pd.read_csv(equipment_file)
        girls_df = df[df["Sport"].isin(GIRLS_SPORTS)]
        spent += girls_df.get("Amount", pd.Series()).sum()
    remaining = GIRLS_BUDGET - spent
    summary = pd.DataFrame({
        "Girls Sports Budget": [GIRLS_BUDGET],
        "Total Spent (Girls Sports)": [spent],
        "Remaining Girls Sports Budget": [remaining]
    })
    return summary

def show_sport_manual(sport):
    manual_file = manual_map.get(sport)
    if manual_file and os.path.exists(manual_file):
        return manual_file
    else:
        return None

def get_coach_athletes(coach_name, password):
    if not os.path.exists(coaches_file):
        return pd.DataFrame(columns=["No coach accounts found."])
    coaches = pd.read_csv(coaches_file)
    row = coaches[coaches['Coach Name'] == coach_name]
    if row.empty:
        return pd.DataFrame(columns=["Login failed: Coach not found."])
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    if row.iloc[0]['Password'] != hashed_pw:
        return pd.DataFrame(columns=["Login failed: Incorrect password."])
    sport = row.iloc[0].get('Sport', None)
    if not sport:
        return pd.DataFrame(columns=["No sport assigned to this coach."])
    if not os.path.exists(csv_file):
        return pd.DataFrame(columns=["No athletes found."])
    athletes = pd.read_csv(csv_file)
    filtered = athletes[athletes['Sport'] == sport]
    def cleared(row):
        checks = [
            bool(row.get("Physical Date")),
            row.get("GPA", 0) >= 2.0,
            bool(row.get("Baseline IMPACT Test Date")),
            bool(row.get("Football Insurance Status")) or bool(row.get("Interscholastic Insurance Status"))
        ]
        return all(checks)
    filtered["Status"] = filtered.apply(lambda r: "Cleared" if cleared(r) else "Not Cleared", axis=1)
    display = filtered[["Name", "Email", "Physical Date", "GPA", "Baseline IMPACT Test Date", "Football Insurance Status", "Interscholastic Insurance Status", "Status"]]
    return display

def get_coach_pay_info(coach_name):
    if not os.path.exists(pay_file):
        return "No pay info available."
    df = pd.read_csv(pay_file)
    row = df[df["Coach Name"].str.lower() == coach_name.strip().lower()]
    if row.empty:
        return "No pay info found for this coach."
    info = row.iloc[0]
    return f"Coach: {info['Coach Name']}\nSport: {info['Sport']}\nSupplement Pay: ${info['Supplement Pay']}\nPay Date: {info['Pay Date']}"

def get_phone_directory():
    if os.path.exists(phone_file):
        return pd.read_csv(phone_file)
    else:
        return pd.DataFrame(columns=["Name", "Role", "Phone Number"])

# --- QR Code for Game Revenue ---
def generate_game_qr(sport, date, opponent, amount):
    qr_data = f"Sport: {sport}\nDate: {date}\nOpponent: {opponent}\nAmount: ${amount}"
    img = qrcode.make(qr_data)
    filename = f"qr_{sport}_{date}_{opponent}.png".replace(" ", "_")
    img.save(filename)
    # Optionally, save a record for revenue tracking
    record = {
        "Sport": sport,
        "Date": date,
        "Opponent": opponent,
        "Amount": amount,
        "QRFile": filename
    }
    if os.path.exists(qr_revenue_file):
        df = pd.read_csv(qr_revenue_file)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    df.to_csv(qr_revenue_file, index=False)
    return filename

# --- QR Code for Staff Entrance (one-time use) ---
def generate_staff_qr(staff_name):
    staff_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    qr_data = f"Staff: {staff_name}\nID: {staff_id}\nTime: {timestamp}"
    img = qrcode.make(qr_data)
    filename = f"staff_qr_{staff_name.replace(' ', '_')}_{staff_id}.png"
    img.save(filename)
    # Save to a log for one-time use validation
    staff_qr_log = "staff_qr_log.csv"
    record = {
        "Staff Name": staff_name,
        "Staff ID": staff_id,
        "Timestamp": timestamp,
        "QRFile": filename,
        "Used": False
    }
    if os.path.exists(staff_qr_log):
        df = pd.read_csv(staff_qr_log)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    df.to_csv(staff_qr_log, index=False)
    return filename

athlete_inputs = [
    gr.Textbox(label="Name"),
    gr.Textbox(label="Date of Birth (YYYY-MM-DD)"),
    gr.Number(label="Age"),
    gr.Textbox(label="School"),
    gr.Dropdown(sports_list, label="Sport"),
    gr.Textbox(label="Position"),
    gr.Textbox(label="Physical Date (YYYY-MM-DD)"),
    gr.Textbox(label="Physical Expiry Date (YYYY-MM-DD)"),
    gr.Number(label="Graduation Year"),
    gr.Textbox(label="9th Grade Entry Date (YYYY-MM-DD)"),
    gr.Textbox(label="Baseline IMPACT Test Date (YYYY-MM-DD)"),
    gr.Textbox(label="Home Phone"),
    gr.Textbox(label="Cell Phone"),
    gr.Textbox(label="Email"),
    gr.Number(label="GPA"),
    gr.Textbox(label="Football Insurance Status"),
    gr.Textbox(label="Interscholastic Insurance Status"),
    gr.File(label="Upload Birth Certificate"),
    gr.File(label="Upload Parent Driver's License")
]

coach_inputs = [
    gr.Textbox(label="Coach Full Name"),
    gr.Textbox(label="Password", type="password"),
    gr.Textbox(label="Confirm Password", type="password"),
    gr.File(label="Upload Coaching Certification"),
    gr.Textbox(label="Certification Expiry Date (YYYY-MM-DD)"),
    gr.Dropdown(sports_list, label="Sport")
]

schedule_inputs = [
    gr.Textbox(label="Date (YYYY-MM-DD)"),
    gr.Textbox(label="Opponent"),
    gr.Dropdown(["Home", "Away"], label="Location"),
    gr.Number(label="Bus Cost (if Away)", value=0),
    gr.Number(label="Referee Pay (if Home)", value=0),
    gr.Number(label="Trainer Pay (if Home)", value=0),
    gr.Number(label="Security Pay (if Home)", value=0),
    gr.Number(label="Other Staff Pay (if Home)", value=0),
    gr.Number(label="Booking Fee (if Home)", value=0)
]

archived_sport_filter = gr.Dropdown(
    ["All"] + sports_list,
    label="Filter by Sport",
    value="All"
)

with gr.Blocks() as demo:
    gr.Markdown(
        """
        <div style='background:orange;padding:10px;border-radius:10px;'>
        <h1 style='color:white;'>Athletes ChatBoy 🧠</h1>
        </div>
        """,
        elem_id="branding"
    )
    with gr.Tab("🏠 Home"):
        gr.Interface(fn=home, inputs=[], outputs="text", live=False)
    with gr.Tab("🗳️ Submit Athlete"):
        gr.Interface(
            fn=submit_athlete,
            inputs=athlete_inputs,
            outputs="text"
        )
    with gr.Tab("🧠 Ask ChatBoy"):
        gr.Interface(
            fn=ask_chatboy,
            inputs=gr.Textbox(label="Ask a question"),
            outputs="text"
        )
    with gr.Tab("📁 View Archived Athletes"):
        gr.Interface(
            fn=view_archived_athletes,
            inputs=archived_sport_filter,
            outputs=gr.Dataframe()
        )
    with gr.Tab("📝 Create Coach Account"):
        gr.Interface(
            fn=create_coach_account,
            inputs=coach_inputs,
            outputs="text"
        )
    with gr.Tab("📅 Sport Schedules"):
        gr.Markdown("### Add New Game to Schedule")
        gr.Interface(
            fn=add_schedule,
            inputs=schedule_inputs,
            outputs="text"
        )
        gr.Markdown("### All Scheduled Games")
        gr.Interface(
            fn=view_schedules,
            inputs=[],
            outputs=gr.Dataframe()
        )
        gr.Markdown("### Cost Summary")
        gr.Interface(
            fn=schedule_summary,
            inputs=[],
            outputs=gr.Dataframe()
        )
        gr.Markdown("### Budget Tracker")
        gr.Interface(
            fn=budget_status,
            inputs=[],
            outputs=gr.Dataframe()
        )
    with gr.Tab("🏋️ Equipment Purchases"):
        gr.Interface(
            fn=add_equipment_purchase,
            inputs=[
                gr.Dropdown(sports_list, label="Sport"),
                gr.Textbox(label="Item"),
                gr.Number(label="Amount ($)")
            ],
            outputs="text"
        )
        gr.Markdown("### All Equipment Purchases")
        gr.Interface(
            fn=lambda: pd.read_csv(equipment_file) if os.path.exists(equipment_file) else pd.DataFrame(columns=["Sport", "Item", "Amount", "Date"]),
            inputs=[],
            outputs=gr.Dataframe()
        )
    with gr.Tab("💊 Trainer Supplies"):
        gr.Interface(
            fn=add_trainer_supply,
            inputs=[
                gr.Textbox(label="Item"),
                gr.Number(label="Amount ($)")
            ],
            outputs="text"
        )
        gr.Markdown("### All Trainer Supply Purchases")
        gr.Interface(
            fn=lambda: pd.read_csv(supplies_file) if os.path.exists(supplies_file) else pd.DataFrame(columns=["Item", "Amount", "Date"]),
            inputs=[],
            outputs=gr.Dataframe()
        )
    with gr.Tab("🎟️ Game QR Code for Revenue"):
        gr.Markdown("### Generate a QR Code for Game Revenue")
        gr.Interface(
            fn=generate_game_qr,
            inputs=[
                gr.Dropdown(sports_list, label="Sport"),
                gr.Textbox(label="Date (YYYY-MM-DD)"),
                gr.Textbox(label="Opponent"),
                gr.Number(label="Amount ($)")
            ],
            outputs=gr.Image(type="filepath")
        )
    with gr.Tab("🛂 Staff Entrance QR Code"):
        gr.Markdown("Generate a one-time QR code for staff entrance.")
        gr.Interface(
            fn=generate_staff_qr,
            inputs=gr.Textbox(label="Staff Name"),
            outputs=gr.Image(type="filepath")
        )
    with gr.Tab("📚 Sports Manual"):
        gr.Markdown("### Download the Manual / Rules & Regulations for Your Sport")
        gr.Interface(
            fn=show_sport_manual,
            inputs=gr.Dropdown(
                list(manual_map.keys()),
                label="Select Sport"
            ),
            outputs=gr.File(label="Download Manual")
        )
    with gr.Tab("📄 Officials Pay Scale"):
        gr.Markdown("### Download the 2024-2025 Officials Pay Scale")
        gr.Interface(
            fn=lambda: "2024-2025_official_pay_scale.pdf" if os.path.exists("2024-2025_official_pay_scale.pdf") else None,
            inputs=[],
            outputs=gr.File(label="Download Pay Scale")
        )
    with gr.Tab("🔒 Coach Login & Athlete Status"):
        gr.Interface(
            fn=get_coach_athletes,
            inputs=[
                gr.Textbox(label="Coach Name"),
                gr.Textbox(label="Password", type="password")
            ],
            outputs=gr.Dataframe(label="Your Athletes")
        )
    with gr.Tab("👧 Girls Sports Budget"):
        gr.Markdown("### Girls Sports Budget Tracker")
        gr.Interface(
            fn=girls_budget_status,
            inputs=[],
            outputs=gr.Dataframe()
        )
    with gr.Tab("💵 Coach Pay Info"):
        gr.Interface(
            fn=get_coach_pay_info,
            inputs=gr.Textbox(label="Enter Your Name"),
            outputs="text"
        )
    with gr.Tab("📞 Phone Directory"):
        gr.Markdown("### Important Contacts")
        gr.Interface(
            fn=get_phone_directory,
            inputs=[],
            outputs=gr.Dataframe()
        )

if __name__ == "__main__":
    demo.launch()