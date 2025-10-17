import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib
import sqlite3
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from plotly import express as px
import plotly.graph_objects as go
import bcrypt

# --- Global Constants ---
DB_NAME = "users.db"
MODEL_PATH = "similarity_model.pkl"
DATA_PATH = "courses_data.csv"
N_COURSES = 300 # Generating 300 courses for robustness

# --- 1. Data Generation and Persistence ---

# Highly accurate domain and skill lists for realistic data
INTEREST_DOMAINS = ['Data Science', 'Web Development', 'Artificial Intelligence', 'Cyber Security', 'Cloud Computing', 'UI/UX Design', 'Finance & Accounting', 'Game Development', 'Digital Marketing']
ALL_SKILLS = ['Python', 'SQL', 'R', 'Pandas', 'NumPy', 'React', 'NodeJs', 'AWS', 'Azure', 'Docker', 'Kubernetes', 'Figma', 'Photoshop', 'JavaScript', 'HTML', 'CSS', 'Scikit-learn', 'TensorFlow', 'Java', 'C++', 'Git', 'SEO', 'PPC', 'Excel', 'Tableau', 'Unity']
DIFFICULTIES = ['Beginner', 'Intermediate', 'Advanced']

@st.cache_resource
def generate_course_data_and_save(n=N_COURSES):
    """Generates synthetic course data and saves it to a CSV."""
    data = []
    
    for i in range(1, n + 1):
        domain = random.choice(INTEREST_DOMAINS)
        platform = random.choice(['Udemy', 'Coursera', 'edX', 'LinkedIn Learning', 'Pluralsight'])
        difficulty = random.choice(DIFFICULTIES)
        
        # Select 3 to 6 skills relevant to the domain
        relevant_skills = [s for s in ALL_SKILLS if s.lower() in domain.lower() or random.random() < 0.3]
        skills_covered = ' '.join(random.sample(relevant_skills, k=min(len(relevant_skills), random.randint(3, 6))))
        
        data.append({
            'course_id': i,
            'course_name': f"Mastering {domain} - Project {i}",
            'category': domain,
            'difficulty_level': difficulty,
            'duration_hours': random.randint(5, 60),
            'rating': round(random.uniform(3.8, 4.9), 2),
            'num_students': random.randint(1000, 150000),
            'skills_covered': skills_covered,
            'price': random.choice([0] * 5 + [9.99, 19.99, 49.99, 99.99, 199.99, 299.99]), # 0 represents Free
            'platform': platform
        })
    
    df = pd.DataFrame(data)
    # df.to_csv(DATA_PATH, index=False) # Uncomment to save file
    return df

# --- 2. Database Functions (SQLite for Auth) ---

def create_user_table():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            interests TEXT
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    # Hash password using bcrypt for security
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def add_user(name, email, password, interests):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        hashed_pw = hash_password(password)
        c.execute("INSERT INTO users (name, email, password, interests) VALUES (?, ?, ?, ?)",
                  (name, email, hashed_pw, interests))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False # User already exists

def login_user(email, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password, name, interests FROM users WHERE email=?", (email,))
    user_data = c.fetchone()
    conn.close()
    
    if user_data and check_password(password, user_data[0]):
        return {'name': user_data[1], 'interests': user_data[2]}
    return None

# --- 3. ML Model Training and Recommendation Logic ---

@st.cache_resource
def train_recommendation_model(df):
    """
    Builds a Hybrid Recommendation Model (Content-based + Popularity).
    """
    # Preprocessing
    df['combined_features'] = df['category'] + ' ' + df['difficulty_level'] + ' ' + df['skills_covered'].fillna('')
    df['combined_features'] = df['combined_features'].str.lower()

    # Content-Based Filtering (TF-IDF on combined features)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Calculate Popularity Score (Weighting students by rating)
    C = df['rating'].mean()
    m = df['num_students'].quantile(0.7) 
    
    def weighted_rating(x, m=m, C=C):
        v = x['num_students']
        R = x['rating']
        return (v/(v+m) * R) + (m/(m+v) * C)

    df['popularity_score'] = df.apply(weighted_rating, axis=1)

    # Save components
    model_components = {
        'cosine_sim': cosine_sim,
        'vectorizer': vectorizer,
        'indices': pd.Series(df.index, index=df['course_name']).drop_duplicates(),
        'df': df
    }
    # joblib.dump(model_components, MODEL_PATH) # Uncomment to save file
    return model_components

def get_hybrid_recommendations(user_input, model_components, df_all):
    """
    Generates recommendations based on user input (Hybrid Approach).
    """
    
    df = model_components['df']
    vectorizer = model_components['vectorizer']
    cosine_sim = model_components['cosine_sim']
    
    # 1. User Profile Vectorization (for content-based score)
    user_profile = (
        f"{user_input['domain']} "
        f"{user_input['difficulty']} "
        f"{user_input['skills']} "
        f"{user_input['goal']} "
    ).lower()
    
    # Transform the user profile using the *fitted* vectorizer
    user_vec = vectorizer.transform([user_profile])
    
    # Calculate similarity between user profile and all courses
    user_course_sim = cosine_similarity(user_vec, vectorizer.transform(df['combined_features'])).flatten()
    
    # 2. Combine with Popularity Score
    df['content_score'] = user_course_sim
    
    # Create Final Recommendation Score (weighted average)
    # Weight Content Score higher for personalization
    df['final_score'] = (0.7 * df['content_score']) + (0.3 * (df['popularity_score'] / 5.0))
    
    # 3. Apply Filters
    filters = (
        (df['difficulty_level'] == user_input['difficulty']) &
        (df['category'].isin([user_input['domain']])) &
        (df['duration_hours'] <= user_input['time_max']) 
    )
    
    if user_input['budget'] == 'Free':
        filters &= (df['price'] == 0)
    elif user_input['budget'] == 'Paid':
        filters &= (df['price'] > 0)

    df_filtered = df[filters].sort_values(by='final_score', ascending=False)
    
    return df_filtered.head(10).drop(columns=['combined_features', 'content_score', 'popularity_score'])

# --- 4. Main Streamlit App Functions ---

def show_login_page():
    st.title("🔐 Welcome to CourseRec AI")
    st.subheader("Login or Create Account")

    choice = st.radio("Select Action", ["Login", "Sign Up"])

    if choice == "Login":
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                user_data = login_user(email, password)
                if user_data:
                    st.session_state.logged_in = True
                    st.session_state.user_name = user_data['name']
                    st.session_state.user_interests = user_data['interests']
                    st.success(f"Welcome back, {st.session_state.user_name}!")
                    st.rerun()
                else:
                    st.error("Invalid Email or Password.")

    else:
        with st.form("signup_form"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            
            # Simple interests field
            interests = st.text_area("Your Learning Interests (e.g., Python, ML, Figma)")

            submitted = st.form_submit_button("Sign Up")

            if submitted:
                if add_user(name, email, password, interests):
                    st.success("Account created successfully! Please log in.")
                else:
                    st.error("User with this email already exists.")

def show_recommendation_page(model_components):
    st.title("🎯 Personalized Course Recommendations")
    st.subheader(f"Hello, {st.session_state.user_name}! Let's find your perfect course.")
    st.markdown("---")
    
    df_all = model_components['df']

    with st.form("recommendation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            domain = st.selectbox("Select Primary Domain/Interest", options=INTEREST_DOMAINS)
            difficulty = st.selectbox("Preferred Difficulty", options=DIFFICULTIES)
            skills = st.text_input("Skills You Already Know (Comma-separated)", value=st.session_state.user_interests)

        with col2:
            goal = st.selectbox("Learning Goal", options=['Job-oriented', 'Skill upgrade', 'Project-based learning'])
            budget = st.selectbox("Budget", options=['Any', 'Free', 'Paid'])
            time_pref = st.select_slider("Preferred Learning Time (Hours)", options=['<10', '10-30', '30-60', '>60'])
        
        submitted = st.form_submit_button("Recommend Courses")

    if submitted:
        time_max = 1000
        if time_pref == '<10': time_max = 10
        elif time_pref == '10-30': time_max = 30
        elif time_pref == '30-60': time_max = 60
        
        user_input = {
            'domain': domain,
            'difficulty': difficulty,
            'skills': skills,
            'goal': goal,
            'budget': budget,
            'time_max': time_max
        }
        
        with st.spinner("Analyzing profile and generating hybrid recommendations..."):
            recommendations = get_hybrid_recommendations(user_input, model_components, df_all)
            
            if recommendations.empty:
                st.warning("No courses match your strict criteria. Try widening your filters!")
            else:
                st.subheader(f"Top {len(recommendations)} Recommendations for You")
                st.write(recommendations)
                
                # Download link (Simplified)
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv = convert_df_to_csv(recommendations)
                st.download_button(
                    label="Download Recommendations as CSV",
                    data=csv,
                    file_name='recommendations.csv',
                    mime='text/csv',
                )

def show_analytics_dashboard(df):
    st.title("📊 Learning Analytics Dashboard")
    st.markdown("Explore trends in the course catalog.")

    # Filter Sidebar
    st.sidebar.header("Analytics Filters")
    platform_filter = st.sidebar.multiselect("Filter by Platform", df['platform'].unique(), default=df['platform'].unique())
    difficulty_filter = st.sidebar.multiselect("Filter by Difficulty", df['difficulty_level'].unique(), default=df['difficulty_level'].unique())
    rating_min = st.sidebar.slider("Minimum Rating", 3.0, 5.0, 4.0)

    df_filtered = df[
        (df['platform'].isin(platform_filter)) &
        (df['difficulty_level'].isin(difficulty_filter)) &
        (df['rating'] >= rating_min)
    ]
    
    st.subheader(f"Displaying {len(df_filtered)} Courses")

    col1, col2 = st.columns(2)

    with col1:
        # Top 10 Most Popular Course Domains (Bar Chart)
        popular_domains = df_filtered.groupby('category')['num_students'].sum().nlargest(10)
        fig_pop = px.bar(popular_domains, x=popular_domains.index, y='num_students', 
                         title='Top 10 Domains by Total Enrollment', labels={'index': 'Domain', 'num_students': 'Total Students'},
                         color=popular_domains.index, color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig_pop, use_container_width=True)

    with col2:
        # Average Rating per Category (Pie Chart)
        avg_rating_cat = df_filtered.groupby('category')['rating'].mean().sort_values(ascending=False).head(8)
        fig_rating = px.pie(avg_rating_cat, names=avg_rating_cat.index, values=avg_rating_cat.values,
                            title='Average Rating per Category (Top 8)', hole=.3)
        st.plotly_chart(fig_rating, use_container_width=True)
        
    # Price Distribution by Platform (Box Plot)
    fig_price = px.box(df_filtered[df_filtered['price'] > 0], x='platform', y='price', 
                       title='Price Distribution by Platform (Paid Courses Only)', 
                       color='platform',
                       color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_price, use_container_width=True)

def show_dataset_explorer(df):
    st.title("📂 Dataset Explorer")
    st.subheader("Course Catalog Overview")

    # Summary Stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Courses", len(df))
    col2.metric("Avg. Rating", f"{df['rating'].mean():.2f} ⭐")
    col3.metric("Unique Domains", df['category'].nunique())

    # Searchable/Sortable Table
    st.dataframe(df, use_container_width=True, height=500)

    # Option to upload new dataset (Simplified, won't actually retrain model here)
    st.subheader("Upload New Data")
    uploaded_file = st.file_uploader("Upload a new CSV dataset", type="csv")
    if uploaded_file is not None:
        st.success("File uploaded successfully! Model retraining feature would be integrated here.")


def main():
    # Session state initialization
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_name = None
        st.session_state.user_interests = ""
        
    create_user_table()
    df_courses = generate_course_data_and_save()
    
    try:
        model_components = train_recommendation_model(df_courses)
    except Exception as e:
        st.error(f"Error loading or training model: {e}")
        st.stop()


    if not st.session_state.logged_in:
        show_login_page()
    else:
        st.sidebar.title(f"Welcome, {st.session_state.user_name}!")
        st.sidebar.markdown("---")

        app_mode = st.sidebar.selectbox("Navigation", 
                                        ["🏠 Dashboard", "🎯 Recommendations", "📊 Analytics", "📂 Data Explorer"])

        if app_mode == "🏠 Dashboard":
            st.title("CourseRec AI Dashboard")
            st.info("Use the sidebar for navigation. Your personalized interests are: " + st.session_state.user_interests)
        elif app_mode == "🎯 Recommendations":
            show_recommendation_page(model_components)
        elif app_mode == "📊 Analytics":
            show_analytics_dashboard(df_courses)
        elif app_mode == "📂 Data Explorer":
            show_dataset_explorer(df_courses)
            
        if st.sidebar.button("Logout", key="logout"):
            st.session_state.logged_in = False
            st.session_state.user_name = None
            st.toast("Logged out successfully.")
            st.rerun()

if __name__ == "__main__":
    main()
