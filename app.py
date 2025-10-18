# Runner Planner - A Streamlit App
# This app helps you track your running data, add new runs, and get performance predictions

# Import required libraries
import streamlit as st  # Main framework for creating web apps
import pandas as pd     # Data manipulation and analysis
import numpy as np      # Numerical computing
from datetime import date, timedelta  # Date handling
from pathlib import Path  # File path operations
import altair as alt    # Advanced charting library

# Import functions from our custom core.py module
from core import (
    DAYS,  # List of weekdays for scheduling
    parse_hms_to_minutes, pace_to_min_per_mile, min_to_pace_str,  # Time/pace conversion helpers
    load_runs, estimate_next_race  # Data loading and prediction functions
)

# Configure the Streamlit page
st.set_page_config(
    page_title="Runner Planner",  # Browser tab title
    page_icon="🏃",               # Browser tab icon
    layout="wide"                 # Use full width of screen
)

# Initialize session state variables to store data between page refreshes
if 'runs_df' not in st.session_state:
    st.session_state.runs_df = None  # Store the loaded running data
if 'generated' not in st.session_state:
    st.session_state.generated = False  # Track if plan has been generated
if 'prediction' not in st.session_state:
    st.session_state.prediction = None  # Store race performance prediction
if 'race_data' not in st.session_state:
    st.session_state.race_data = {}  # Store race information

def normalize_dataframe(df):
    """
    Normalize the DataFrame to consistent column names
    Handles naming inconsistencies and ensures proper data types
    """
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    df_norm = df.copy()
    
    # Ensure date is datetime and sorted first
    if 'date' in df_norm.columns:
        df_norm['date'] = pd.to_datetime(df_norm['date'])
        df_norm = df_norm.sort_values('date').reset_index(drop=True)
    
    # Create elapsed_min from duration if not present
    if 'elapsed_min' not in df_norm.columns and 'duration' in df_norm.columns:
        df_norm['elapsed_min'] = df_norm['duration'].apply(parse_hms_to_minutes)
    
    # Normalize distance column names
    if 'distance_miles' in df_norm.columns:
        df_norm['distance_mi'] = df_norm['distance_miles']
    elif 'distance_mi' not in df_norm.columns and 'distance' in df_norm.columns:
        df_norm['distance_mi'] = df_norm['distance']
    
    # Normalize pace column names
    if 'pace_min_per_mile' in df_norm.columns:
        df_norm['pace_min_per_mi'] = df_norm['pace_min_per_mile']
    elif 'pace_min_per_mi' not in df_norm.columns and 'average pace' in df_norm.columns:
        df_norm['pace_min_per_mi'] = df_norm['average pace'].apply(pace_to_min_per_mile)
    
    # Calculate pace from distance and duration if pace is missing
    if 'pace_min_per_mi' not in df_norm.columns or df_norm['pace_min_per_mi'].isna().all():
        if 'elapsed_min' in df_norm.columns and 'distance_mi' in df_norm.columns:
            df_norm['pace_min_per_mi'] = df_norm['elapsed_min'] / df_norm['distance_mi']
    
    # Clean up infinite values
    df_norm = df_norm.replace([np.inf, -np.inf], np.nan)
    
    # Only drop rows if the required columns exist
    required_cols = []
    if 'date' in df_norm.columns:
        required_cols.append('date')
    if 'distance_mi' in df_norm.columns:
        required_cols.append('distance_mi')
    if 'pace_min_per_mi' in df_norm.columns:
        required_cols.append('pace_min_per_mi')
    
    if required_cols:
        df_norm = df_norm.dropna(subset=required_cols)
    
    return df_norm

def load_initial_data():
    """
    Load initial data from data/running_data.csv or create empty DataFrame
    """
    default_path = Path("data/running_data.csv")
    
    if default_path.exists():
        try:
            # Try loading with core.py function first
            df = load_runs(
                path=default_path,
                date_col='date',
                dist_col='distance',
                time_col='duration',
                pace_col='average pace',
                unit='miles'
            )
            if df is not None and not df.empty:
                return normalize_dataframe(df)
        except Exception as e:
            st.warning(f"Could not load with core.py: {e}")
            
        try:
            # Fallback: load CSV directly and normalize
            df = pd.read_csv(default_path)
            if not df.empty:
                return normalize_dataframe(df)
        except Exception as e:
            st.warning(f"Could not load CSV directly: {e}")
    
    # Create empty DataFrame with correct columns if no data found
    return pd.DataFrame(columns=[
        'date', 'distance', 'average pace', 'duration', 'average heart rate', 
        'average heart rate zone', 'run type', 'shoe gear'
    ])

def save_data_to_csv(df, filepath="data/running_data.csv"):
    """
    Save DataFrame to CSV file with proper error handling
    """
    try:
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False

def add_new_run(run_data):
    """
    Add a new run to the dataset and save to CSV
    """
    try:
        # Create new row with the run data
        new_run = pd.DataFrame([run_data])
        
        # Add to existing data
        if st.session_state.runs_df is not None and not st.session_state.runs_df.empty:
            updated_df = pd.concat([st.session_state.runs_df, new_run], ignore_index=True)
        else:
            updated_df = new_run
        
        # Sort by date
        updated_df = updated_df.sort_values('date').reset_index(drop=True)
        
        # Save to CSV
        if save_data_to_csv(updated_df):
            # Update session state
            st.session_state.runs_df = updated_df
            return True
        else:
            return False
            
    except Exception as e:
        st.error(f"Error adding run: {e}")
        return False

def main():
    # Display the main title and description
    st.title("🏃 Runner Planner")
    st.markdown("Track your runs, add new entries, and get performance predictions.")
    
    # Load initial data if not already loaded
    if st.session_state.runs_df is None:
        st.session_state.runs_df = load_initial_data()
    
    # Sidebar for data input
    with st.sidebar:
        st.header("📁 Data Input")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload your running data CSV", 
            type=['csv'],
            help="Upload a CSV file with your running data"
        )
        
        # File path input
        file_path = st.text_input(
            "Or enter file path:", 
            value="data/running_data.csv",
            help="Path to your CSV file"
        )
        
        # Load data button
        if st.button("🔄 Load Data", type="primary"):
            if uploaded_file is not None:
                # Save uploaded file temporarily
                temp_path = Path("temp_upload.csv")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    # Try loading with core.py function first
                    df = load_runs(
                        path=temp_path,
                        date_col='date',
                        dist_col='distance',
                        time_col='duration',
                        pace_col='average pace',
                        unit='miles'
                    )
                    if df is not None and not df.empty:
                        st.session_state.runs_df = normalize_dataframe(df)
                        st.success(f"✅ Loaded {len(st.session_state.runs_df)} runs successfully!")
                    else:
                        st.error("No valid data found.")
                except Exception as e:
                    st.warning(f"Could not load with core.py: {e}")
                    try:
                        # Fallback: load CSV directly
                        df = pd.read_csv(temp_path)
                        if not df.empty:
                            st.session_state.runs_df = normalize_dataframe(df)
                            st.success(f"✅ Loaded {len(st.session_state.runs_df)} runs successfully!")
                        else:
                            st.error("No valid data found.")
                    except Exception as e2:
                        st.error(f"Error loading data: {str(e2)}")
                finally:
                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()
            else:
                # Try file path
                data_path = Path(file_path)
                if data_path.exists():
                    try:
                        # Try loading with core.py function first
                        df = load_runs(
                            path=data_path,
                            date_col='date',
                            dist_col='distance',
                            time_col='duration',
                            pace_col='average pace',
                            unit='miles'
                        )
                        if df is not None and not df.empty:
                            st.session_state.runs_df = normalize_dataframe(df)
                            st.success(f"✅ Loaded {len(st.session_state.runs_df)} runs successfully!")
                        else:
                            st.error("No valid data found.")
                    except Exception as e:
                        st.warning(f"Could not load with core.py: {e}")
                        try:
                            # Fallback: load CSV directly
                            df = pd.read_csv(data_path)
                            if not df.empty:
                                st.session_state.runs_df = normalize_dataframe(df)
                                st.success(f"✅ Loaded {len(st.session_state.runs_df)} runs successfully!")
                            else:
                                st.error("No valid data found.")
                        except Exception as e2:
                            st.error(f"Error loading data: {str(e2)}")
                    else:
                        st.warning(f"File not found: {data_path}")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🏃 Your Runs", "📊 Visualize Data", "🎯 Performance Prediction"])
    
    # TAB 1: Your Runs (Add new entries and data table)
    with tab1:
        col1, col2 = st.columns([2, 1])  # Main content and sidebar for race form
        
        with col1:
            # Add Run Form
            with st.expander("➕ Add a new run", expanded=False):
                st.subheader("Run Details")
                
                # Create form columns
                col1a, col1b = st.columns(2)
                
                with col1a:
                    run_date = st.date_input("Date", value=date.today())
                    distance = st.number_input("Distance (miles)", min_value=0.1, max_value=100.0, step=0.1)
                    avg_pace = st.text_input("Average pace (MM:SS)", placeholder="09:00")
                    duration = st.text_input("Duration (HH:MM:SS)", placeholder="1:30:00")
                
                with col1b:
                    avg_hr = st.number_input("Average heart rate", min_value=0, max_value=300, value=0, step=1)
                    hr_zone = st.text_input("Average heart rate zone", placeholder="Zone 3")
                    run_type = st.selectbox("Run type", ["Easy", "Tempo", "Intervals", "Long", "Recovery", "Race"])
                    shoe_gear = st.text_input("Shoe gear", placeholder="Nike Air Zoom")
                
                # Submit button
                if st.button("🏃 Add Run", type="primary"):
                    # Validate inputs
                    if distance <= 0:
                        st.error("Distance must be greater than 0")
                    elif not avg_pace:
                        st.error("Please enter average pace")
                    else:
                        # Convert pace to numeric
                        pace_min_per_mi = pace_to_min_per_mile(avg_pace)
                        if pd.isna(pace_min_per_mi):
                            st.error("Invalid pace format. Use MM:SS (e.g., 09:00)")
                        else:
                            # Convert duration to numeric
                            elapsed_min = parse_hms_to_minutes(duration) if duration else None
                            if elapsed_min is None and duration:
                                st.error("Invalid duration format. Use HH:MM:SS or MM:SS")
                            else:
                                # Create run data
                                run_data = {
                                    'date': run_date,
                                    'distance': distance,
                                    'average pace': avg_pace,
                                    'duration': duration if duration else "",
                                    'average heart rate': avg_hr if avg_hr > 0 else "",
                                    'average heart rate zone': hr_zone if hr_zone else "",
                                    'run type': run_type,
                                    'shoe gear': shoe_gear if shoe_gear else ""
                                }
                                
                                # Add run to dataset
                                if add_new_run(run_data):
                                    st.success("✅ Run added successfully!")
                                    st.rerun()  # Refresh the app to show updated data
                                else:
                                    st.error("Failed to add run")
        
        with col2:
            # Add Race Form
            with st.expander("🏁 Add Race Details", expanded=False):
                st.subheader("Race Information")
                
                race_name = st.text_input("Race name", placeholder="Boston Marathon")
                race_date = st.date_input("Race date", value=date.today() + timedelta(weeks=8))
                
                # Distance selector
                distance_option = st.radio(
                    "Race distance",
                    ["5K (3.1 miles)", "10K (6.2 miles)", "Half Marathon (13.1 miles)", "Marathon (26.2 miles)"],
                    index=2  # Default to Half Marathon
                )
                
                # Map distance option to miles
                distance_mapping = {
                    "5K (3.1 miles)": 3.1069,
                    "10K (6.2 miles)": 6.2137,
                    "Half Marathon (13.1 miles)": 13.1,
                    "Marathon (26.2 miles)": 26.2
                }
                race_distance = distance_mapping[distance_option]
                
                goal_pace = st.text_input("Goal pace (MM:SS)", placeholder="09:00")
                
                # Save race data
                if st.button("💾 Save Race", type="primary"):
                    st.session_state.race_data = {
                        'name': race_name,
                        'date': race_date,
                        'distance': race_distance,
                        'goal_pace': goal_pace
                    }
                    st.success("✅ Race details saved!")
        
        # Data Table Section
        st.header("📋 Your Running Data")
        
        if st.session_state.runs_df is not None and not st.session_state.runs_df.empty:
            # Display the data table
            st.dataframe(
                st.session_state.runs_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv_data = st.session_state.runs_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Running Data as CSV",
                data=csv_data,
                file_name=f"running_data_{date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Show summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Runs", len(st.session_state.runs_df))
            
            with col2:
                if 'distance_mi' in st.session_state.runs_df.columns:
                    total_miles = st.session_state.runs_df['distance_mi'].sum()
                else:
                    total_miles = st.session_state.runs_df['distance'].sum()
                st.metric("Total Miles", f"{total_miles:.0f}")
            
            with col3:
                if 'pace_min_per_mi' in st.session_state.runs_df.columns:
                    avg_pace = st.session_state.runs_df['pace_min_per_mi'].mean()
                else:
                    # Convert average pace to numeric
                    pace_numeric = st.session_state.runs_df['average pace'].apply(pace_to_min_per_mile)
                    avg_pace = pace_numeric.mean()
                st.metric("Average Pace", min_to_pace_str(avg_pace))
            
            with col4:
                if 'average heart rate' in st.session_state.runs_df.columns:
                    avg_hr = st.session_state.runs_df['average heart rate'].mean()
                    if not pd.isna(avg_hr) and avg_hr > 0:
                        st.metric("Average HR", f"{avg_hr:.0f}")
                    else:
                        st.metric("Average HR", "—")
                else:
                    st.metric("Average HR", "—")
        else:
            st.info("👆 Add your first run to get started!")
    
    # TAB 2: Visualize Data (Charts)
    with tab2:
        st.header("📊 Training Progress")
        
        if st.session_state.runs_df is not None and not st.session_state.runs_df.empty:
            # Check if we have the required columns before creating charts
            has_distance = 'distance_mi' in st.session_state.runs_df.columns or 'distance' in st.session_state.runs_df.columns
            has_pace = 'pace_min_per_mi' in st.session_state.runs_df.columns or 'average pace' in st.session_state.runs_df.columns
            
            if has_distance:
                # Distance over time
                st.subheader("Distance Over Time")
                if 'distance_mi' in st.session_state.runs_df.columns:
                    distance_chart = st.session_state.runs_df[['date', 'distance_mi']].copy()
                else:
                    distance_chart = st.session_state.runs_df[['date', 'distance']].copy()
                st.line_chart(distance_chart.set_index('date'))
            
            if has_pace:
                # Pace over time
                st.subheader("Pace Over Time (Lower is Faster)")
                if 'pace_min_per_mi' in st.session_state.runs_df.columns:
                    pace_chart = st.session_state.runs_df[['date', 'pace_min_per_mi']].copy()
                else:
                    # Convert average pace to numeric for charting
                    pace_numeric = st.session_state.runs_df['average pace'].apply(pace_to_min_per_mile)
                    pace_chart = st.session_state.runs_df[['date']].copy()
                    pace_chart['pace_min_per_mi'] = pace_numeric
                st.line_chart(pace_chart.set_index('date'))
            
            # Weekly mileage
            st.subheader("Weekly Mileage")
            if 'distance_mi' in st.session_state.runs_df.columns:
                weekly_data = st.session_state.runs_df.set_index('date').resample('W-MON')['distance_mi'].sum().reset_index()
            else:
                weekly_data = st.session_state.runs_df.set_index('date').resample('W-MON')['distance'].sum().reset_index()
            weekly_data.columns = ['Week', 'Miles']
            if len(weekly_data) > 0:
                st.bar_chart(weekly_data.set_index('Week'))
            else:
                st.info("No weekly data available.")
        else:
            st.info("👆 Add some runs to see your training progress!")
    
    # TAB 3: Performance Prediction
    with tab3:
        st.header("🎯 Performance Prediction")
        
        if st.session_state.runs_df is not None and not st.session_state.runs_df.empty:
            # Distance selector for prediction
            pred_distance_option = st.radio(
                "Select distance for prediction:",
                ["5K (3.1 miles)", "10K (6.2 miles)", "Half Marathon (13.1 miles)", "Marathon (26.2 miles)"],
                horizontal=True,
                index=2  # Default to Half Marathon
            )
            
            # Map distance option to miles
            distance_mapping = {
                "5K (3.1 miles)": 3.1069,
                "10K (6.2 miles)": 6.2137,
                "Half Marathon (13.1 miles)": 13.1,
                "Marathon (26.2 miles)": 26.2
            }
            pred_distance = distance_mapping[pred_distance_option]
            
            # Get goal pace from race data if available
            goal_pace_str = ""
            if st.session_state.race_data and 'goal_pace' in st.session_state.race_data:
                goal_pace_str = st.session_state.race_data['goal_pace']
            
            # Generate prediction
            if st.button("🔮 Generate Prediction", type="primary"):
                try:
                    # Prepare data for prediction (rename columns to match core.py expectations)
                    runs_for_pred = st.session_state.runs_df.copy()
                    
                    # Ensure we have the required columns for prediction
                    if 'distance_mi' in runs_for_pred.columns:
                        runs_for_pred['distance_miles'] = runs_for_pred['distance_mi']
                    elif 'distance' in runs_for_pred.columns:
                        runs_for_pred['distance_miles'] = runs_for_pred['distance']
                    
                    if 'pace_min_per_mi' in runs_for_pred.columns:
                        runs_for_pred['pace_min_per_mile'] = runs_for_pred['pace_min_per_mi']
                    elif 'average pace' in runs_for_pred.columns:
                        # Convert average pace to numeric
                        runs_for_pred['pace_min_per_mile'] = runs_for_pred['average pace'].apply(pace_to_min_per_mile)
                    
                    # Call prediction function
                    prediction = estimate_next_race(
                        runs=runs_for_pred,
                        goal_distance_miles=pred_distance,
                        goal_pace_str=goal_pace_str
                    )
                    
                    st.session_state.prediction = prediction
                    st.success("✅ Prediction generated!")
                    
                except Exception as e:
                    st.error(f"Error generating prediction: {str(e)}")
            
            # Display prediction results
            if st.session_state.prediction is not None:
                st.subheader("Race Prediction")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Suggested Race Pace",
                        st.session_state.prediction.get('suggested_pace_str', '—')
                    )
                
                with col2:
                    st.metric(
                        "Estimated Finish Time",
                        st.session_state.prediction.get('pred_time_str', '—')
                    )
        else:
            st.info("👆 Add some runs to get performance predictions!")

# This runs the main function when the script is executed
if __name__ == "__main__":
    main()