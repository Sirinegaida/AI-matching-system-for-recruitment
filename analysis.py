import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
import traceback
from datetime import datetime


def add_chart_export_options(fig, chart_name):
    """Add export options for charts"""
    with st.expander(f"📥 Export {chart_name}"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(f"📊 PNG", key=f"png_{chart_name}"):
                try:
                    img_bytes = fig.to_image(format="png", width=1200, height=800)
                    st.download_button(
                        label="⬇️ Download PNG",
                        data=img_bytes,
                        file_name=f"{chart_name.lower().replace(' ', '_')}.png",
                        mime="image/png",
                        key=f"download_png_{chart_name}"
                    )
                except Exception as e:
                    st.error(f"PNG export failed: {e}")
        
        with col2:
            if st.button(f"📋 HTML", key=f"html_{chart_name}"):
                try:
                    html_str = fig.to_html()
                    st.download_button(
                        label="⬇️ Download HTML",
                        data=html_str,
                        file_name=f"{chart_name.lower().replace(' ', '_')}.html",
                        mime="text/html",
                        key=f"download_html_{chart_name}"
                    )
                except Exception as e:
                    st.error(f"HTML export failed: {e}")
        
        with col3:
            st.info("💡 Export charts for reports")


def demonstrate_chart_libraries(db_manager):
    """Demonstrate different chart libraries with real CV data"""
    st.subheader("📈 Chart Libraries Comparison")
    
    try:
        # Get sample data
        analytics = db_manager.get_skills_analytics()
        
        if not analytics or 'technical_skills' not in analytics or analytics['technical_skills'].empty:
            st.warning("No data available for demonstration")
            return
        
        sample_data = analytics['technical_skills'].head(10)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Plotly", "📈 Matplotlib", "🎨 Altair", "📋 Streamlit Native"])
        
        with tab1:
            st.write("**Plotly Charts (Interactive & Professional)**")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_plotly_bar = px.bar(sample_data, x='skill', y='frequency', 
                                      title="Plotly Bar Chart")
                fig_plotly_bar.update_xaxes(tickangle=45)
                st.plotly_chart(fig_plotly_bar, use_container_width=True)
            
            with col2:
                fig_plotly_pie = px.pie(sample_data, values='frequency', names='skill',
                                      title="Plotly Pie Chart")
                st.plotly_chart(fig_plotly_pie, use_container_width=True)
        
        with tab2:
            st.write("**Matplotlib Charts (Static & Customizable)**")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar chart
            ax1.bar(sample_data['skill'], sample_data['frequency'])
            ax1.set_title('Matplotlib Bar Chart')
            ax1.set_xlabel('Skills')
            ax1.set_ylabel('Frequency')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Pie chart
            ax2.pie(sample_data['frequency'], labels=sample_data['skill'], autopct='%1.1f%%')
            ax2.set_title('Matplotlib Pie Chart')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)  # Close figure to prevent memory leaks
        
        with tab3:
            st.write("**Altair Charts (Grammar of Graphics)**")
            
            # Altair bar chart
            chart_bar = alt.Chart(sample_data).mark_bar().encode(
                x=alt.X('skill:N', axis=alt.Axis(labelAngle=45)),
                y='frequency:Q',
                color=alt.Color('frequency:Q', scale=alt.Scale(scheme='viridis'))
            ).properties(
                title='Altair Bar Chart',
                width=400,
                height=300
            )
            st.altair_chart(chart_bar, use_container_width=True)
        
        with tab4:
            st.write("**Streamlit Native Charts (Simple & Fast)**")
            
            # Prepare data for streamlit charts
            chart_data = sample_data.set_index('skill')
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Bar Chart")
                st.bar_chart(chart_data['frequency'])
            
            with col2:
                st.write("Line Chart")
                st.line_chart(chart_data['frequency'])
    
    except Exception as e:
        st.error(f"Error in chart libraries demonstration: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")



def create_skills_correlation_matrix(db_manager):
    """Create skills correlation and co-occurrence analysis"""
    st.subheader("🔗 Skills Correlation & Co-occurrence Analysis")
    
    try:
        # Get skills data
        skills_query = """
        SELECT skills
        FROM candidates 
        WHERE skills IS NOT NULL
        """
        df = db_manager.execute_query_safely(skills_query)
        
        if df.empty:
            st.info("No skills data available for correlation analysis")
            return
        
        # Parse skills and create co-occurrence matrix
        all_tech_skills = []
        valid_skills_count = 0
        
        for skills_json in df['skills']:
            if skills_json:
                try:
                    skills_dict = json.loads(skills_json) if isinstance(skills_json, str) else skills_json
                    tech_skills = skills_dict.get('technical_skills', [])
                    if tech_skills:
                        all_tech_skills.extend(tech_skills)
                        valid_skills_count += 1
                except (json.JSONDecodeError, TypeError):
                    continue
        
        if not all_tech_skills:
            st.info("No technical skills found for correlation analysis")
            return
        
        # Get top skills for correlation analysis
        from collections import Counter
        skill_counts = Counter(all_tech_skills)
        top_skills = [skill for skill, count in skill_counts.most_common(15)]
        
        if len(top_skills) > 5:
            # Create co-occurrence matrix
            cooccurrence_matrix = np.zeros((len(top_skills), len(top_skills)))
            
            for skills_json in df['skills']:
                if skills_json:
                    try:
                        skills_dict = json.loads(skills_json) if isinstance(skills_json, str) else skills_json
                        candidate_skills = skills_dict.get('technical_skills', [])
                        
                        for i, skill1 in enumerate(top_skills):
                            for j, skill2 in enumerate(top_skills):
                                if skill1 in candidate_skills and skill2 in candidate_skills:
                                    cooccurrence_matrix[i][j] += 1
                    except (json.JSONDecodeError, TypeError):
                        continue
            
            # Create heatmap
            fig_corr = px.imshow(
                cooccurrence_matrix,
                labels=dict(x="Skills", y="Skills", color="Co-occurrence"),
                x=top_skills,
                y=top_skills,
                title="Skills Co-occurrence Matrix",
                color_continuous_scale='RdBu'
            )
            fig_corr.update_xaxes(tickangle=45)
            fig_corr.update_yaxes(tickangle=0)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Skills network visualization (simplified)
            st.info("💡 Skills that frequently appear together suggest complementary skill sets")
        else:
            st.info("Not enough diverse skills for correlation analysis")
        
    except Exception as e:
        st.error(f"Error in skills correlation analysis: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")


def create_advanced_skills_dashboard(db_manager):
    """Create an advanced skills dashboard with multiple chart types"""
    
    st.header("📊 Advanced Skills Analytics Dashboard")
    
    try:
        # Get analytics data
        analytics = db_manager.get_skills_analytics()
        
        if not analytics:
            st.info("No data available for analytics")
            return
        
        # === 1. SKILLS COMPARISON CHART ===
        if 'technical_skills' in analytics and 'soft_skills' in analytics:
            st.subheader("🔄 Technical vs Soft Skills Comparison")
            
            tech_total = analytics['technical_skills']['frequency'].sum() if not analytics['technical_skills'].empty else 0
            soft_total = analytics['soft_skills']['frequency'].sum() if not analytics['soft_skills'].empty else 0
            
            if tech_total > 0 or soft_total > 0:
                # Create comparison pie chart
                comparison_data = pd.DataFrame({
                    'Skill Type': ['Technical Skills', 'Soft Skills'],
                    'Total Mentions': [tech_total, soft_total]
                })
                
                fig_comparison = px.pie(
                    comparison_data,
                    values='Total Mentions',
                    names='Skill Type',
                    title="Technical vs Soft Skills Distribution",
                    color_discrete_map={'Technical Skills': '#FF6B6B', 'Soft Skills': '#4ECDC4'}
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Add export option
                add_chart_export_options(fig_comparison, "Skills Comparison")
        
        # === 2. SKILLS HEATMAP ===
        if 'technical_skills' in analytics and not analytics['technical_skills'].empty:
            st.subheader("🔥 Top Skills Heatmap")
            
            top_skills = analytics['technical_skills'].head(15)
            
            if len(top_skills) > 0:
                # Create heatmap data (simulated proficiency levels)
                skills_matrix = np.random.rand(4, len(top_skills))
                skill_categories = ['Beginner', 'Intermediate', 'Advanced', 'Expert']
                
                fig_heatmap = px.imshow(
                    skills_matrix,
                    labels=dict(x="Skills", y="Proficiency Level", color="Frequency"),
                    x=top_skills['skill'].tolist(),
                    y=skill_categories,
                    title="Skills Proficiency Heatmap",
                    color_continuous_scale='Viridis'
                )
                fig_heatmap.update_xaxes(tickangle=45)
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                add_chart_export_options(fig_heatmap, "Skills Heatmap")
        
        # === 3. GEOGRAPHIC DISTRIBUTION ===
        st.subheader("🌍 Candidate Geographic Distribution")
        
        try:
            # Get geographic data from database
            geo_query = """
            SELECT country, city, COUNT(*) as candidate_count
            FROM candidates 
            WHERE country IS NOT NULL AND country != ''
            GROUP BY country, city
            ORDER BY candidate_count DESC;
            """
            geo_df = db_manager.execute_query_safely(geo_query)
            
            if not geo_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Country distribution pie chart
                    country_counts = geo_df.groupby('country')['candidate_count'].sum().reset_index()
                    
                    fig_geo = px.pie(
                        country_counts,
                        values='candidate_count',
                        names='country',
                        title="Candidates by Country"
                    )
                    st.plotly_chart(fig_geo, use_container_width=True)
                
                with col2:
                    # City distribution bar chart
                    top_cities = geo_df.head(10)
                    fig_cities = px.bar(
                        top_cities,
                        x='candidate_count',
                        y='city',
                        orientation='h',
                        title="Top 10 Cities by Candidate Count",
                        color='candidate_count',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_cities, use_container_width=True)
                
                # World map visualization
                if len(country_counts) > 1:
                    fig_world = px.choropleth(
                        country_counts,
                        locations='country',
                        locationmode='country names',
                        color='candidate_count',
                        hover_name='country',
                        title="Global Candidate Distribution",
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_world, use_container_width=True)
            else:
                st.info("No geographic data available")
                
        except Exception as e:
            st.error(f"Error loading geographic data: {e}")
        
        # === 4. EXPERIENCE VS EDUCATION ANALYSIS ===
        st.subheader("🎓 Experience vs Education Analysis")
        
        try:
            exp_edu_query = """
            SELECT experience_years, education_level, job_category, 
                   first_name, last_name, COUNT(*) as count
            FROM candidates 
            WHERE experience_years IS NOT NULL AND education_level IS NOT NULL
            GROUP BY experience_years, education_level, job_category, first_name, last_name;
            """
            exp_edu_df = db_manager.execute_query_safely(exp_edu_query)
            
            if not exp_edu_df.empty:
                # Aggregate for scatter plot
                scatter_data = exp_edu_df.groupby(['experience_years', 'education_level', 'job_category']).size().reset_index(name='count')
                
                fig_scatter = px.scatter(
                    scatter_data,
                    x='experience_years',
                    y='education_level',
                    size='count',
                    color='job_category',
                    title="Experience Years vs Education Level",
                    hover_data=['count'],
                    size_max=20
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Box plot for experience distribution by education
                fig_box = px.box(
                    exp_edu_df,
                    x='education_level',
                    y='experience_years',
                    title="Experience Distribution by Education Level"
                )
                fig_box.update_xaxes(tickangle=45)
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No experience/education data available")
                
        except Exception as e:
            st.error(f"Error loading experience/education data: {e}")
    
    except Exception as e:
        st.error(f"Error in advanced skills dashboard: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")


def create_skills_word_cloud_simulation(db_manager):
    """Simulate a word cloud effect using bubble charts"""
    st.subheader("☁️ Skills Bubble Cloud")
    
    try:
        analytics = db_manager.get_skills_analytics()
        
        if not analytics or 'technical_skills' not in analytics or analytics['technical_skills'].empty:
            st.info("No technical skills data available for bubble cloud")
            return
        
        skills_df = analytics['technical_skills'].head(30).copy()
        
        if len(skills_df) > 0:
            # Add random positions for bubble effect
            skills_df['x'] = np.random.rand(len(skills_df)) * 100
            skills_df['y'] = np.random.rand(len(skills_df)) * 100
            
            fig_cloud = px.scatter(
                skills_df,
                x='x',
                y='y',
                size='frequency',
                text='skill',
                title="Technical Skills Bubble Cloud",
                size_max=50,
                color='frequency',
                color_continuous_scale='viridis'
            )
            
            fig_cloud.update_traces(textposition="middle center", textfont_size=10)
            fig_cloud.update_layout(
                showlegend=False,
                xaxis={'visible': False, 'showticklabels': False},
                yaxis={'visible': False, 'showticklabels': False}
            )
            
            st.plotly_chart(fig_cloud, use_container_width=True)
        else:
            st.info("No skills data available for bubble cloud")
    
    except Exception as e:
        st.error(f"Error creating skills bubble cloud: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")


def create_advanced_experience_analysis(db_manager):
    """Advanced experience-based analytics"""
    st.subheader("💼 Advanced Experience Analysis")
    
    try:
        exp_query = """
        SELECT experience_years, job_category, education_level, 
               COUNT(*) as count
        FROM candidates 
        WHERE experience_years IS NOT NULL
        GROUP BY experience_years, job_category, education_level
        ORDER BY experience_years;
        """
        exp_df = db_manager.execute_query_safely(exp_query)
        
        if exp_df.empty:
            st.info("No experience data available for analysis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Experience distribution as bar chart
            exp_summary = exp_df.groupby('experience_years')['count'].sum().reset_index()
            
            fig_hist = px.bar(
                exp_summary,
                x='experience_years',
                y='count',
                title="Experience Distribution",
                labels={'experience_years': 'Years of Experience', 'count': 'Number of Candidates'},
                color='count',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col2:
            # Experience by category violin plot
            # Create expanded dataset for violin plot
            exp_detailed = []
            for _, row in exp_df.iterrows():
                for _ in range(row['count']):
                    exp_detailed.append({
                        'experience_years': row['experience_years'],
                        'job_category': row['job_category']
                    })
            exp_detailed_df = pd.DataFrame(exp_detailed)
            
            if not exp_detailed_df.empty and len(exp_detailed_df['job_category'].unique()) > 1:
                fig_violin = px.violin(
                    exp_detailed_df,
                    x='job_category',
                    y='experience_years',
                    title="Experience Distribution by Job Category",
                    box=True
                )
                fig_violin.update_xaxes(tickangle=45)
                st.plotly_chart(fig_violin, use_container_width=True)
            else:
                # Fallback to box plot if violin plot doesn't work
                if not exp_detailed_df.empty:
                    fig_box = px.box(
                        exp_detailed_df,
                        x='job_category',
                        y='experience_years',
                        title="Experience Range by Job Category"
                    )
                    fig_box.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("Not enough data for category comparison")
        
        # Experience statistics
        st.subheader("📊 Experience Statistics")
        
        # Calculate statistics from aggregated data
        total_candidates = int(exp_df['count'].sum()) if not exp_df.empty else 0
        
        if total_candidates > 0:
            weighted_avg = (exp_df['experience_years'] * exp_df['count']).sum() / total_candidates
            max_exp = int(exp_df['experience_years'].max())
            min_exp = int(exp_df['experience_years'].min())
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("👥 Total Candidates", f"{total_candidates:,}")
            with col2:
                st.metric("📊 Average Experience", f"{weighted_avg:.1f} years")
            with col3:
                st.metric("📈 Maximum Experience", f"{max_exp} years")
            with col4:
                st.metric("📉 Minimum Experience", f"{min_exp} years")
        else:
            st.info("No experience data available for statistics")
        
        # 3D scatter plot (if we have enough diverse data)
        if len(exp_df) > 10 and len(exp_df['job_category'].unique()) > 2:
            fig_3d = px.scatter_3d(
                exp_df,
                x='experience_years',
                y='job_category',
                z='education_level',
                size='count',
                color='job_category',
                title="3D Experience-Category-Education Analysis",
                size_max=20,
                hover_data=['count']
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            # Alternative 2D analysis
            st.subheader("📈 Experience vs Education Level")
            
            exp_edu_summary = exp_df.groupby(['experience_years', 'education_level'])['count'].sum().reset_index()
            
            if not exp_edu_summary.empty:
                fig_scatter = px.scatter(
                    exp_edu_summary,
                    x='experience_years',
                    y='education_level',
                    size='count',
                    title="Experience vs Education Level",
                    hover_data=['count'],
                    size_max=25
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Experience progression analysis
        st.subheader("📈 Experience Progression by Category")
        
        if len(exp_df['job_category'].unique()) > 1:
            category_exp = exp_df.groupby(['job_category', 'experience_years'])['count'].sum().reset_index()
            
            fig_line = px.line(
                category_exp,
                x='experience_years',
                y='count',
                color='job_category',
                title="Experience Distribution Trends by Job Category",
                markers=True
            )
            st.plotly_chart(fig_line, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error in experience analysis: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")


def create_timeline_analytics(db_manager):
    """Create timeline-based analytics"""
    st.subheader("📈 Timeline Analytics")
    
    try:
        timeline_query = """
        SELECT DATE(created_at) as date, COUNT(*) as daily_count,
               job_category
        FROM candidates 
        WHERE created_at IS NOT NULL
        GROUP BY DATE(created_at), job_category
        ORDER BY date;
        """
        
        timeline_df = db_manager.execute_query_safely(timeline_query)
        
        if timeline_df.empty:
            st.info("No timeline data available (candidates may not have created_at dates)")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily registrations line chart
            daily_total = timeline_df.groupby('date')['daily_count'].sum().reset_index()
            fig_timeline = px.line(
                daily_total,
                x='date',
                y='daily_count',
                title="Daily Candidate Registrations",
                markers=True
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col2:
            # Cumulative registrations
            daily_total['cumulative'] = daily_total['daily_count'].cumsum()
            fig_cumulative = px.area(
                daily_total,
                x='date',
                y='cumulative',
                title="Cumulative Candidate Registrations"
            )
            st.plotly_chart(fig_cumulative, use_container_width=True)
        
        # Category timeline
        if 'job_category' in timeline_df.columns and len(timeline_df['job_category'].unique()) > 1:
            fig_category_timeline = px.line(
                timeline_df,
                x='date',
                y='daily_count',
                color='job_category',
                title="Registration Timeline by Job Category"
            )
            st.plotly_chart(fig_category_timeline, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading timeline data: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")