import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from io import BytesIO

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Define color palette
COLORS = {
    'primary': '#3B82F6',  # Blue
    'secondary': '#8B5CF6',  # Purple
    'accent': '#F97316',  # Orange
    'success': '#22C55E',  # Green
    'warning': '#EAB308',  # Yellow
    'error': '#EF4444',  # Red
    'neutral': '#64748B'  # Slate
}

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def create_distribution_plot(data, column='Sleep_Duration'):
    if column not in data.columns:
        print(f"{column} not found, using dummy data")
        data = pd.DataFrame({column: np.random.normal(7, 1.5, 100)})

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[column], kde=True, color=COLORS['primary'], bins=20, ax=ax)

    ax.set_title(f'Distribution of {column.replace("_", " ")}')
    ax.set_xlabel(column.replace("_", " "))
    ax.set_ylabel('Frequency')

    plt.tight_layout()
    img_str = fig_to_base64(fig)
    plt.close(fig)
    return img_str

def create_sleep_duration_plot(data):
    """
    Create a visualization of sleep duration by occupation
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed sleep data
        
    Returns:
    --------
    str
        Base64 encoded image
    """
    # Check if required columns exist
    required_cols = ['Occupation', 'Sleep_Duration']
    if not all(col in data.columns for col in required_cols):
        # Use dummy data if columns not found
        print("Required columns not found, using dummy data")
        occupations = ['Healthcare', 'Engineer', 'Teacher', 'Accountant', 'Lawyer']
        sleep_durations = [6.7, 7.2, 6.9, 7.5, 6.5]
        dummy_data = pd.DataFrame({
            'Occupation': occupations,
            'Sleep_Duration': sleep_durations
        })
        data = dummy_data
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create visualization
    occupation_sleep = data.groupby('Occupation')['Sleep_Duration'].mean().sort_values(ascending=False)
    
    # Bar plot
    sns.barplot(x=occupation_sleep.index, y=occupation_sleep.values, 
                palette=sns.color_palette([COLORS['primary'], COLORS['secondary']]*10), ax=ax)
    
    # Add horizontal line for average sleep duration
    avg_sleep = data['Sleep_Duration'].mean()
    ax.axhline(y=avg_sleep, color=COLORS['error'], linestyle='--', 
               label=f'Average: {avg_sleep:.2f} hours')
    
    # Add labels and title
    ax.set_xlabel('Occupation')
    ax.set_ylabel('Average Sleep Duration (hours)')
    ax.set_title('Average Sleep Duration by Occupation')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add legend and adjust layout
    plt.legend()
    plt.tight_layout()
    
    # Convert to base64
    img_str = fig_to_base64(fig)
    plt.close(fig)
    
    return img_str

def create_sleep_quality_plot(data):
    """
    Create a visualization of sleep quality by occupation
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed sleep data
        
    Returns:
    --------
    str
        Base64 encoded image
    """
    # Check if required columns exist
    required_cols = ['Occupation', 'Quality_of_Sleep']
    if not all(col in data.columns for col in required_cols):
        # Use dummy data if columns not found
        print("Required columns not found, using dummy data")
        occupations = ['Healthcare', 'Engineer', 'Teacher', 'Accountant', 'Lawyer']
        sleep_quality = [7.1, 6.8, 7.5, 6.9, 6.2]
        dummy_data = pd.DataFrame({
            'Occupation': occupations,
            'Quality_of_Sleep': sleep_quality
        })
        data = dummy_data
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create visualization
    occupation_quality = data.groupby('Occupation')['Quality_of_Sleep'].mean().sort_values(ascending=False)
    
    # Bar plot
    bars = sns.barplot(x=occupation_quality.index, y=occupation_quality.values, 
                palette=sns.color_palette([COLORS['secondary'], COLORS['primary']]*10), ax=ax)
    
    # Add labels and title
    ax.set_xlabel('Occupation')
    ax.set_ylabel('Average Sleep Quality (1-10)')
    ax.set_title('Average Sleep Quality by Occupation')
    
    # Add value labels on bars
    for i, bar in enumerate(bars.patches):
        bars.text(bar.get_x() + bar.get_width()/2.,
                 bar.get_height() + 0.1,
                 f'{occupation_quality.values[i]:.1f}',
                 ha='center', va='bottom')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert to base64
    img_str = fig_to_base64(fig)
    plt.close(fig)
    
    return img_str

def create_physical_activity_plot(data):
    """
    Create a visualization showing the influence of physical activity on sleep quality
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed sleep data
        
    Returns:
    --------
    str
        Base64 encoded image
    """
    # Check if required columns exist
    required_cols = ['Physical_Activity', 'Quality_of_Sleep']
    if not all(col in data.columns for col in required_cols):
        # Use dummy data if columns not found
        print("Required columns not found, using dummy data")
        activity_levels = [30, 45, 60, 75, 90, 15, 20, 25, 35, 40]
        sleep_quality = [6.5, 7.2, 8.1, 8.5, 8.7, 5.8, 6.2, 6.7, 7.0, 7.3]
        dummy_data = pd.DataFrame({
            'Physical_Activity': activity_levels,
            'Quality_of_Sleep': sleep_quality
        })
        data = dummy_data
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create visualization - Scatter plot with regression line
    sns.regplot(x='Physical_Activity', y='Quality_of_Sleep', data=data, 
                scatter_kws={'alpha':0.5, 'color':COLORS['primary']},
                line_kws={'color':COLORS['accent']}, ax=ax)
    
    # Add labels and title
    ax.set_xlabel('Physical Activity (minutes per day)')
    ax.set_ylabel('Sleep Quality Rating (1-10)')
    ax.set_title('Relationship Between Physical Activity and Sleep Quality')
    
    # Calculate correlation
    correlation = data['Physical_Activity'].corr(data['Quality_of_Sleep'])
    
    # Add correlation annotation
    ax.annotate(f"Correlation: {correlation:.2f}", 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS['neutral'], alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert to base64
    img_str = fig_to_base64(fig)
    plt.close(fig)
    
    return img_str

def create_age_sleep_quality_plot(data):
    """
    Create a box + strip plot showing the relationship between age groups and sleep quality.
    """
    print("Plot created successfully")

    # Normalize column names to avoid KeyErrors
    print("Creating Age vs Sleep Quality plot")
    data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
    print("Cleaned Columns:", data.columns.tolist())


    # Required columns
    required_cols = ['Age', 'Quality_of_Sleep']
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = data[required_cols].dropna()

    # Create age groups
    bins = [18, 30, 40, 50, 60, 100]
    labels = ['18-29', '30-39', '40-49', '50-59', '60+']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Age_Group', y='Quality_of_Sleep', data=df, palette='coolwarm', ax=ax)
    sns.stripplot(x='Age_Group', y='Quality_of_Sleep', data=df, color='black', alpha=0.3, ax=ax)

    ax.set_title('Sleep Quality Distribution by Age Group')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Quality of Sleep (1–10)')
    plt.tight_layout()

    return fig_to_base64(fig)


# def create_age_sleep_quality_plot(data):
#     """
#     Create a visualization showing the relationship between age and sleep quality
    
#     Parameters:
#     -----------
#     data : pandas.DataFrame
#         Processed sleep data
        
#     Returns:
#     --------
#     str
#         Base64 encoded image
#     """
#     # Check if required columns exist
#     required_cols = ['Age', 'Quality_of_Sleep']
#     data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
#     if 'Age' not in data.columns or 'Quality_of_Sleep' not in data.columns:
#         raise ValueError("Required columns missing")

#     df = data[['Age', 'Quality_of_Sleep']].dropna()

#     bins = [18, 30, 40, 50, 60, 100]
#     labels = ['18-29', '30-39', '40-49', '50-59', '60+']
#     df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.boxplot(x='Age_Group', y='Quality_of_Sleep', data=df, palette='coolwarm', ax=ax)
#     sns.stripplot(x='Age_Group', y='Quality_of_Sleep', data=df, color='black', alpha=0.3, ax=ax)

#     ax.set_title('Sleep Quality Distribution by Age Group')
#     ax.set_xlabel('Age Group')
#     ax.set_ylabel('Quality of Sleep (1-10)')
#     plt.tight_layout()
#     return fig_to_base64(fig)
#     # if not all(col in data.columns for col in required_cols):
#     #     # Use dummy data if columns not found
#     #     print("Required columns not found, using dummy data")
#     #     ages = np.random.randint(18, 80, 100)
#     #     sleep_quality = np.clip(8 - (ages-30)/20 + np.random.normal(0, 1, 100), 1, 10)
#     #     dummy_data = pd.DataFrame({
#     #         'Age': ages,
#     #         'Quality_of_Sleep': sleep_quality
#     #     })
#     #     data = dummy_data
    
#     # # Create figure
#     # fig, ax = plt.subplots(figsize=(10, 6))
    
#     # # Create age groups
#     # bins = [18, 30, 40, 50, 60, 100]
#     # labels = ['18-29', '30-39', '40-49', '50-59', '60+']
#     # data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
    
#     # # Create box plot
#     # sns.boxplot(x='Age_Group', y='Quality_of_Sleep', data=data, 
#     #             palette=[COLORS['primary'], COLORS['secondary'], 
#     #                      COLORS['accent'], COLORS['success'], COLORS['warning']], ax=ax)
    
#     # # Add individual points for better visualization
#     # sns.stripplot(x='Age_Group', y='Quality_of_Sleep', data=data, 
#     #               size=4, color=COLORS['neutral'], alpha=0.3, ax=ax)
    
#     # # Add labels and title
#     # ax.set_xlabel('Age Group')
#     # ax.set_ylabel('Sleep Quality Rating (1-10)')
#     # ax.set_title('Sleep Quality Distribution by Age Group')
    
#     # # Adjust layout
#     # plt.tight_layout()
    
#     # # Convert to base64
#     # img_str = fig_to_base64(fig)
#     # plt.close(fig)
    
#     # return img_str

def create_sleep_disorder_comparison_plot(data):
    """
    Create a visualization comparing sleep metrics by disorder type
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed sleep data
        
    Returns:
    --------
    str
        Base64 encoded image
    """
    # Check if required columns exist
    required_cols = ['Sleep_Disorder', 'Sleep_Duration', 'Quality_of_Sleep']
    if not all(col in data.columns for col in required_cols):
        # Use dummy data if columns not found
        print("Required columns not found, using dummy data")
        disorders = ['None', 'Insomnia', 'Sleep Apnea', 'None', 'Insomnia', 'Sleep Apnea', 
                    'None', 'None', 'Insomnia', 'Sleep Apnea']
        sleep_duration = [7.5, 5.2, 6.1, 7.8, 5.5, 6.3, 7.2, 7.6, 5.0, 6.4]
        sleep_quality = [8.2, 4.5, 5.8, 7.9, 4.8, 5.5, 8.0, 7.8, 4.2, 5.9]
        dummy_data = pd.DataFrame({
            'Sleep_Disorder': disorders,
            'Sleep_Duration': sleep_duration,
            'Quality_of_Sleep': sleep_quality
        })
        data = dummy_data
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Fill NaN values in Sleep_Disorder with 'None'
    data['Sleep_Disorder'] = data['Sleep_Disorder'].fillna('None')
    
    # Sleep Duration by Disorder
    sns.barplot(x='Sleep_Disorder', y='Sleep_Duration', data=data, 
                palette=[COLORS['success'], COLORS['error'], COLORS['warning']], ax=ax1)
    ax1.set_title('Sleep Duration by Disorder')
    ax1.set_xlabel('Sleep Disorder')
    ax1.set_ylabel('Average Sleep Duration (hours)')
    
    # Sleep Quality by Disorder
    sns.barplot(x='Sleep_Disorder', y='Quality_of_Sleep', data=data, 
                palette=[COLORS['success'], COLORS['error'], COLORS['warning']], ax=ax2)
    ax2.set_title('Sleep Quality by Disorder')
    ax2.set_xlabel('Sleep Disorder')
    ax2.set_ylabel('Average Sleep Quality (1-10)')
    
    # Add overall title and adjust layout
    plt.suptitle('Impact of Sleep Disorders on Sleep Metrics', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Convert to base64
    img_str = fig_to_base64(fig)
    plt.close(fig)
    
    return img_str
##   ++++++
def generate_plot(plot_type, data, y_true=None, y_pred=None):
    if plot_type == 'sleep_duration':
        return create_sleep_duration_plot(data)
    elif plot_type == 'sleep_quality':
        return create_sleep_quality_plot(data)
    elif plot_type == 'physical_activity':
        return create_physical_activity_plot(data)
    elif plot_type == 'age_sleep_quality':
        return create_age_sleep_quality_plot(data)
    elif plot_type == 'sleep_disorder_comparison':
        return create_sleep_disorder_comparison_plot(data)
    elif plot_type == 'distribution':
        return create_distribution_plot(data, column='Sleep_Duration')
    elif plot_type == 'stress_distribution':
        return create_distribution_plot(data, column='Stress_Level')
    elif plot_type == 'confusion_matrix':
        return create_confusion_matrix_plot(y_true, y_pred)
    elif plot_type == 'correlation_heatmap':
        return create_correlation_heatmap(data)
    elif plot_type == 'scree_plot':
        return create_scree_plot(data)
    elif plot_type == 'pair_plot':
        return create_pair_plot(data)
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")


# Add this to models/visualizations.py or top of your Flask app file
def create_sleep_disorder_distribution_plot(data):
    disorder_counts = data['Sleep_Disorder'].value_counts()
    
    fig, ax = plt.subplots()
    ax.pie(disorder_counts, labels=disorder_counts.index, autopct='%1.1f%%')
    ax.set_title("Sleep Disorder Distribution")
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    
    return image_str

def create_confusion_matrix_plot(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    return fig_to_base64(plt.gcf())

def create_correlation_heatmap(data):
    corr = data.select_dtypes(include='number').corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    return fig_to_base64(fig)

def create_scree_plot(data):
    numeric_data = data.select_dtypes(include='number').dropna()
    pca = PCA()
    pca.fit(numeric_data)
    explained_variance = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', color=COLORS['primary'])
    ax.set_title('Scree Plot (PCA)')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    plt.tight_layout()
    return fig_to_base64(fig)


def create_pair_plot(data):
    # Standardize column names to avoid KeyError
    data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')

    required_cols = ['Age', 'Sleep_Duration', 'Quality_of_Sleep', 'Stress_Level']
    missing = [col for col in required_cols if col not in data.columns]

    if missing:
        raise KeyError(f"Missing columns for pair plot: {missing}")

    subset = data[required_cols].dropna()

    pair = sns.pairplot(subset, diag_kind='kde', palette='husl')

    buf = io.BytesIO()
    pair.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return img_str
