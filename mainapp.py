import streamlit as st
from apps import license_plate_detection
from apps import speed_and_tracking
from apps import parking_detection
from apps import accident_detection

# Set page config
st.set_page_config(page_title="RoadSage", page_icon="logo.jpg", layout="wide")

# Custom CSS to adjust spacing, font size, and text color
st.markdown(
    """
    <style>
    .stSelectbox > div > div:hover {
        cursor: pointer;
    }
    .main-title {
        margin-top: 0px;
        display: flex;
        align-items: center;
    }
    .main-title img {
        margin-right: 0px;
    }
    .main-title p{
        margin-top:25px;
        color:gray;
    }
    .stSidebar > div > div > div > label {
        color: black !important;
    }
    .stSidebar > div > div > div > div > div > div {
        color: black !important;
    }
    .block-container {
        padding: 45px;
        padding-left: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    # Create a title section with logo and heading
    col1, col2 = st.columns([1, 10])
    
    with col1:
        st.image("logo.jpg", width=80)
    
    with col2:
        st.markdown("""
            <div class="main-title" style='color:black'>
                <font size="8"><b>RoadSage : </b></font> <p><font size="6"> Intelligent Traffic Monitoring System</font></p>
            </div>
            """, unsafe_allow_html=True)
        

    # Sidebar with selection box and descriptions
    st.sidebar.title("Select Functionality")
    option = st.sidebar.selectbox(
        "Choose one:", 
        ["Select an option", "License Plate Detection", "Speed and Tracking", "Parking Detection", "Accident Detection"],
        index=0
    )

    st.sidebar.markdown("### Descriptions")
    st.sidebar.markdown("**License Plate Detection:** Detects license plates from videos.", unsafe_allow_html=True)
    st.sidebar.markdown("**Speed and Tracking:** Monitors speed and tracks number of vehicles.", unsafe_allow_html=True)
    st.sidebar.markdown("**Parking Detection:** Detects parking spots and monitors availability.", unsafe_allow_html=True)
    st.sidebar.markdown("**Accident Detection:** Identifies potential accidents or collisions in videos.", unsafe_allow_html=True)

    # Unique key for file uploader based on selected option
    uploader_key = option.replace(" ", "_").lower()

    # Redirect based on selection
    if option == "License Plate Detection":
        license_plate_detection.run(uploader_key)
    elif option == "Speed and Tracking":
        speed_and_tracking.run(uploader_key)
    elif option == "Parking Detection":
        parking_detection.run(uploader_key)
    elif option == "Accident Detection":
        accident_detection.run(uploader_key)
    else:
        col1, col2 = st.columns([1, 1])
    
        with col1:
            st.image("pagepic.jpg", width=380)
        
        with col2:
            st.markdown("""
                <div class="main-title" style='color:black'>
                    <p><font size="4">Welcome to RoadSage, an intelligent traffic monitoring system designed to streamline parking, enhance safety, and reduce congestion around our institution. Using advanced license plate detection, speed monitoring, and accident detection, RoadSage ensures a safer, more efficient campus environment. Choose a functionality to explore how we transform traffic management.</font></p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
