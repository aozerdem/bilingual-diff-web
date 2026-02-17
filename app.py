import pandas as pd
import matplotlib.pyplot as plt

# ... existing content above line 356

# Updated code block

with c1:
    st.markdown("#### Edit Intensity Distribution")
    st.caption("How heavy were the changes?")
    
    # Prepare Data
    categories = list(stats["graph_data"].keys())
    counts = list(stats["graph_data"].values())
    
    # Create Matplotlib Figure
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(categories, counts, color=["#4caf50", "#ff9800", "#f44336"])
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("Count")
    plt.xticks(rotation=15, ha='right')
    
    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    st.pyplot(fig)