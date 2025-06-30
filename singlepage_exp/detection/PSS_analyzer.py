import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import json
from collections import defaultdict

faster_ann = '/home/mserrao/PSSComics/CoMix/data/predicts.coco/DCM/faster_mix/all.json'
magi_ann = '/home/mserrao/PSSComics/CoMix/data/predicts.coco/DCM/magi/val.json'

CLS_MAPPING = {
    1: 'Panel',
    2: 'Character',
    4: 'Text',
    7: 'Face'
}

def load_data(path = faster_ann):
    with open(path, 'r') as f:
        data = json.load(f)
    
    images_df = pd.DataFrame(data['images'])
    annotations_df = pd.DataFrame(data['annotations'])

    images_df['book_hash'] = images_df['file_name'].apply(lambda x: x.split('/')[0])
    
    image_to_book = dict(zip(images_df['id'], images_df['book_hash']))
    
    annotations_df['book_hash'] = annotations_df['image_id'].map(image_to_book)

    unique_image_ids = annotations_df['image_id'].unique()
    image_id_to_page = {img_id: page_num for page_num, img_id in enumerate(unique_image_ids, start=0)}
    annotations_df['page_number'] = annotations_df['image_id'].map(image_id_to_page)
    
    book_page_mapping = {}
    for book in annotations_df['book_hash'].unique():
        book_images = annotations_df[annotations_df['book_hash'] == book]['image_id'].unique()
        for i, img_id in enumerate(sorted(book_images)):
            book_page_mapping[(book, img_id)] = i
    
    annotations_df['page_number_book'] = annotations_df.apply(
        lambda row: book_page_mapping.get((row['book_hash'], row['image_id']), -1), 
        axis=1
    )
    
    images_df['page_number'] = images_df['id'].map(image_id_to_page)
    images_df['page_number_book'] = images_df.apply(
        lambda row: book_page_mapping.get((row['book_hash'], row['id']), -1),
        axis=1
    )

    return images_df, annotations_df

def max_bbox_size(bbox):
    if isinstance(bbox, list) and len(bbox) >= 4:
        width = bbox[2]
        height = bbox[3]
        return max(width, height)
    return 0

def calculate_bbox_centroid(bbox):
    if isinstance(bbox, list) and len(bbox) >= 4:
        x = bbox[0] + bbox[2] / 2
        y = bbox[1] + bbox[3] / 2
        return (x, y)
    return (0, 0)

def page_stream_analysis(annotations_df):
    
    annotations_df['bbox_max_dim'] = annotations_df['bbox'].apply(max_bbox_size)
    
    count_per_page_category = annotations_df.groupby(['page_number', 'category_id']).size().reset_index(name='bbox_count')
    
    page_stats = (
        annotations_df
        .groupby(['page_number', 'category_id', 'page_number_book'])
        .agg({
            'bbox_max_dim': ['max', 'mean'],
            'area': ['max', 'mean']
        })
        .reset_index()
    )
    
    page_stats.columns = ['_'.join(col).strip('_') for col in page_stats.columns.values]
    
    page_stats = page_stats.merge(
        count_per_page_category, 
        on=['page_number', 'category_id']
    )

    return page_stats

def calculate_advanced_features(annotations_df, images_df):
    """
    Calculate additional advanced features:
    - Maximum area panel centroid position
    - Panel Coverage
    - Text to panel ratio
    - Character to text ratio
    - Centroid of the biggest text box
    """
    annotations_df['bbox_width'] = annotations_df['bbox'].apply(lambda x: x[2] if isinstance(x, list) and len(x) >= 4 else 0)
    annotations_df['bbox_height'] = annotations_df['bbox'].apply(lambda x: x[3] if isinstance(x, list) and len(x) >= 4 else 0)
    annotations_df['bbox_max_dim'] = annotations_df['bbox'].apply(max_bbox_size)
    annotations_df['bbox_centroid'] = annotations_df['bbox'].apply(calculate_bbox_centroid)
    annotations_df['bbox_centroid_x'] = annotations_df['bbox_centroid'].apply(lambda x: x[0])
    annotations_df['bbox_centroid_y'] = annotations_df['bbox_centroid'].apply(lambda x: x[1])
    
    page_dimensions = {}
    for _, row in images_df.iterrows():
        page_dimensions[row['id']] = (row.get('width', 1000), row.get('height', 1500)) 
    
    annotations_df['page_width'] = annotations_df['image_id'].map(lambda x: page_dimensions.get(x, (1000, 1500))[0])
    annotations_df['page_height'] = annotations_df['image_id'].map(lambda x: page_dimensions.get(x, (1000, 1500))[1])
    annotations_df['page_area'] = annotations_df['page_width'] * annotations_df['page_height']
    
    result = []
    
    for page_id in annotations_df['image_id'].unique():
        page_data = annotations_df[annotations_df['image_id'] == page_id]
        
        page_width, page_height = page_dimensions.get(page_id, (1000, 1500))
        page_area = page_width * page_height
        
        book_hash = page_data['book_hash'].iloc[0] if not page_data.empty else "unknown"
        page_number = page_data['page_number'].iloc[0] if not page_data.empty else -1
        page_number_book = page_data['page_number_book'].iloc[0] if not page_data.empty else -1
        
        panels = page_data[page_data['category_id'] == 1]
        characters = page_data[page_data['category_id'] == 2]
        texts = page_data[page_data['category_id'] == 4]
        faces = page_data[page_data['category_id'] == 7]

        max_panel_centroid_x = 0.5
        max_panel_centroid_y = 0.5
        
        if not panels.empty:
            max_area_panel = panels.loc[panels['area'].idxmax()] if not panels.empty else None
            if max_area_panel is not None:
                max_panel_centroid = calculate_bbox_centroid(max_area_panel['bbox'])
                max_panel_centroid_x = max_panel_centroid[0] / page_width  
                max_panel_centroid_y = max_panel_centroid[1] / page_height  
        
        total_panel_area = panels['area'].sum() if not panels.empty else 0
        panel_coverage = total_panel_area / page_area if page_area > 0 else 0
        
        total_text_area = texts['area'].sum() if not texts.empty else 0
        text_to_panel_ratio = total_text_area / total_panel_area if total_panel_area > 0 else 0
        
        total_character_area = characters['area'].sum() if not characters.empty else 0
        character_to_text_ratio = total_character_area / total_text_area if total_text_area > 0 else 0
        
        max_text_centroid_x = 0.5
        max_text_centroid_y = 0.5
        
        if not texts.empty:
            max_area_text = texts.loc[texts['area'].idxmax()] if not texts.empty else None
            if max_area_text is not None:
                max_text_centroid = calculate_bbox_centroid(max_area_text['bbox'])
                max_text_centroid_x = max_text_centroid[0] / page_width  
                max_text_centroid_y = max_text_centroid[1] / page_height  
        
        features = {
            'image_id': page_id,
            'book_hash': book_hash,
            'page_number': page_number,
            'page_number_book': page_number_book,
            'page_width': page_width,
            'page_height': page_height,
            'page_area': page_area,
            
            'panel_count': len(panels),
            'character_count': len(characters),
            'text_count': len(texts),
            'face_count': len(faces),
            
            'total_panel_area': total_panel_area,
            'total_character_area': total_character_area,
            'total_text_area': total_text_area,
            'total_face_area': faces['area'].sum() if not faces.empty else 0,
            
            'max_panel_centroid_x': max_panel_centroid_x,
            'max_panel_centroid_y': max_panel_centroid_y,
            'panel_coverage': panel_coverage,
            'text_to_panel_ratio': text_to_panel_ratio,
            'character_to_text_ratio': character_to_text_ratio,
            'max_text_centroid_x': max_text_centroid_x,
            'max_text_centroid_y': max_text_centroid_y,
        }
        
        result.append(features)
    
    return pd.DataFrame(result)

def plot_page_stream_segmentation(page_stats, output_path=None):
    
    categories = page_stats['category_id'].unique()
    
    metrics = [
        ('bbox_max_dim_max', 'Maximum Bounding Box Dimension'),
        ('area_max', 'Maximum Bbox Area'),
        ('area_mean', 'Mean Area per Class'),
        ('bbox_count', 'Bbox Count per Class')  
    ]
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    category_colors = {
        1: 'blue',      # Panel
        2: 'green',     # Character
        4: 'red',       # Text
        7: 'purple'     # Face
    }
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        
        for category in categories:
            category_data = page_stats[page_stats['category_id'] == category]
            
            category_data = category_data.sort_values('page_number')

            ax.plot(
                category_data['page_number'], 
                category_data[metric],
                marker='o', 
                linestyle='-', 
                label=f"{CLS_MAPPING.get(category, f'Category {category}')}",
                color=category_colors.get(category, 'gray'),
                alpha=0.8
            )
            
            if idx == 0: 
                ax_right = ax.twinx()
                for category in categories:
                    category_data = page_stats[page_stats['category_id'] == category]
                    if not category_data.empty and len(category_data) > 1:
                        
                        sns.kdeplot(
                            y=category_data[metric], 
                            ax=ax_right, 
                            color=category_colors.get(category, 'gray'),
                            alpha=0.2
                        )
                        
                ax_right.set_yticks([])
                ax_right.set_ylabel('Density')
        
        ax.set_title(f'Page Stream Analysis: {title}')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Page Number')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig, axes

def plot_book_stats(page_stats, top_k_features=None, output_path=None, legend_position='default'):
    """
    Plot book statistics with configurable legend placement.
    
    Args:
        page_stats: DataFrame containing page statistics
        top_k_features: List of metrics to display
        output_path: Where to save the output figure
        legend_position: Where to place legend ('outside', 'bottom', or 'default')
    """
    categories = page_stats['category_id'].unique()
    
    if top_k_features is None:
        metrics = [
            ('bbox_max_dim_max', 'Maximum Bounding Box Dimension'),
            ('area_max', 'Maximum Bbox Area'),
            ('area_mean', 'Mean Area per Class'),
            ('bbox_count', 'Bbox Count per Class')  
        ]
    else:
        metrics = top_k_features
    
    # Add extra space at bottom if legend is at bottom
    if legend_position == 'bottom':
        fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 12), sharex=True)
        plt.subplots_adjust(hspace=0.3, bottom=0.2)  # Extra space at bottom
    else:
        fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 12), sharex=True)
        plt.subplots_adjust(hspace=0.3)
    
    if len(metrics) == 1:
        axes = [axes]
    
    category_colors = {
        1: "#0088FF",      # Panel
        2: "#4136BD",     # Character
        4: "#5E70C9",       # Text
        7: "#C7E3FB"     # Face
    }

    page_types = page_stats['page_type'].unique()
    page_type_colors = {
        'story': "#9FD2FF",           # Light blue
        'cover': "#7D9CDE",           # Light green
        'advertisement': "#6B46B779",   # Light yellow
        'textstory': "#1D3B62",       # Light red
        'story_first_page': "#408DAB" # Light lime
    }
    
    unique_pages = page_stats[['page_number', 'page_number_book', 'page_type']].drop_duplicates()
    unique_pages = unique_pages.sort_values(by='page_number_book')
    
    x_column = 'page_number_book'
    
    hash = page_stats['book_hash'].unique()[0]
    print(hash)
    
    # Create lists to store legend handles and labels
    all_category_handles = []
    all_category_labels = []
    all_page_type_handles = []
    all_page_type_labels = []
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        
        if metric not in page_stats.columns:
            ax.text(0.5, 0.5, f"Metric '{metric}' not found in dataset", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            continue
        
        prev_page_type = None
        start_x = None
        
        for i, (_, page_row) in enumerate(unique_pages.iterrows()):
            page_num = page_row[x_column]
            page_type = page_row['page_type']
        
            if page_type != prev_page_type:
                if prev_page_type is not None:
                    end_x = page_num
                    ax.axvspan(start_x-0.5, end_x-0.5, alpha=0.2, 
                              color=page_type_colors.get(prev_page_type, 'lightgray'))
                prev_page_type = page_type
                start_x = page_num
                  
            if i == len(unique_pages) - 1:
                ax.axvspan(start_x-0.5, page_num+0.5, alpha=0.2, 
                          color=page_type_colors.get(page_type, 'lightgray'))
        
        for category in categories:
            category_data = page_stats[page_stats['category_id'] == category]
            category_data = category_data.sort_values(x_column)
            
            if len(category_data) > 0 and metric in category_data.columns:
                line = ax.plot(
                    category_data[x_column], 
                    category_data[metric],
                    marker='o', 
                    linestyle='-', 
                    label=f"{CLS_MAPPING.get(category, f'Category {category}')}",
                    color=category_colors.get(category, 'gray'),
                    alpha=0.8
                )
                
                # Only store handles once
                if idx == 0:
                    all_category_handles.append(line[0])
                    all_category_labels.append(f"{CLS_MAPPING.get(category, f'Category {category}')}")
        
        if idx == 0: 
            ax_right = ax.twinx()
            for category in categories:
                category_data = page_stats[page_stats['category_id'] == category]
                if not category_data.empty and len(category_data) > 1 and metric in category_data.columns:
                    sns.kdeplot(
                        y=category_data[metric], 
                        ax=ax_right, 
                        color=category_colors.get(category, 'gray'),
                        alpha=0.2
                    )
            ax_right.set_yticks([])
            ax_right.set_ylabel('Density')
        
        for page_type in page_types:
            type_pages = unique_pages[unique_pages['page_type'] == page_type]
            if not type_pages.empty:
                y_max = ax.get_ylim()[1]
                marker = ax.scatter(
                    type_pages[x_column], 
                    [y_max * 0.95] * len(type_pages),
                    marker='|', 
                    s=100, 
                    color=page_type_colors.get(page_type, 'gray'),
                    label=f"Page Type: {page_type}" 
                )
                
                # Only store handles once
                if idx == 0:
                    all_page_type_handles.append(marker)
                    all_page_type_labels.append(f"Page Type: {page_type}")
        
        ax.set_title(f'Page Stream Analysis: {title} Hash:{hash}')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Page Number')
    
    # Handle legend placement
    if legend_position == 'outside':
        # Place legends outside the plot
        if all_category_handles:
            fig.legend(all_category_handles, all_category_labels, 
                      loc='center left', bbox_to_anchor=(1.0, 0.5), title='Categories')
        
        if all_page_type_handles:
            fig.legend(all_page_type_handles, all_page_type_labels, 
                     loc='center left', bbox_to_anchor=(1.15, 0.5), title='Page Types')
            
    elif legend_position == 'bottom':
        # Place legends at the bottom
        if all_category_handles and all_page_type_handles:
            combined_handles = all_category_handles + all_page_type_handles
            combined_labels = all_category_labels + all_page_type_labels
            
            fig.legend(combined_handles, combined_labels, 
                      loc='upper center', bbox_to_anchor=(0.5, 0.05),
                      fancybox=True, shadow=True, ncol=min(5, len(combined_handles)))
    else:
        # Default legend placement inside the first axes
        if all_category_handles:
            category_legend = axes[0].legend(all_category_handles, all_category_labels, 
                                          loc='upper left', title='Categories')
            axes[0].add_artist(category_legend)
          
        if all_page_type_handles:
            axes[0].legend(all_page_type_handles, all_page_type_labels, 
                         loc='upper right', title='Page Types')
    
    # Adjust layout and return
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig, axes

def analyze_comic_book(annots_path, output_dir=None, plot=True):
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    images_df, annotations_df = load_data(annots_path)
    
    page_stats = page_stream_analysis(annotations_df)
    
    if plot:
        basic_fig, _ = plot_page_stream_segmentation(page_stats)
        if output_dir:
            basic_fig.savefig(f"{output_dir}/page_stream_metrics.png", dpi=300, bbox_inches='tight')
            
        
    return {
        'images': images_df,
        'annotations': annotations_df,
        'page_stats': page_stats,
    }

    
def detect_pages_by_thresholds(page_stats, thresholds):
    """
    Detect pages that meet specific thresholds for different metrics and classes.
    
    Parameters:
    -----------
    page_stats : DataFrame
        DataFrame containing page statistics as generated by page_stream_analysis()
    thresholds : dict
        Dictionary with format:
        {
            (category_id, metric): (comparison_operator, threshold_value)
        }
        where:
        - category_id is the class ID (1:Panel, 2:Character, 4:Text, 7:Face)
        - metric is the column name in page_stats (e.g., 'area_mean', 'bbox_count')
        - comparison_operator is a string, one of: '>', '>=', '<', '<=', '=='
        - threshold_value is the value to compare against
    
    Returns:
    --------
    dict
        Dictionary with page numbers as keys and lists of conditions met as values
    """
    detected_pages = defaultdict(list)
    
    for page_number in page_stats['page_number'].unique():
        page_data = page_stats[page_stats['page_number'] == page_number]
        
        for (cat_id, metric), (operator, value) in thresholds.items():

            cat_row = page_data[page_data['category_id'] == cat_id]

            if cat_row.empty:
                continue
      
            actual_value = cat_row[metric].iloc[0]

            condition_met = False
            if operator == '>':
                condition_met = actual_value > value
            elif operator == '>=':
                condition_met = actual_value >= value
            elif operator == '<':
                condition_met = actual_value < value
            elif operator == '<=':
                condition_met = actual_value <= value
            elif operator == '==':
                condition_met = actual_value == value

            if condition_met:
                condition_desc = f"{CLS_MAPPING.get(cat_id, f'Category {cat_id}')} {metric} {operator} {value}"
                detected_pages[page_number].append(condition_desc)

    all_conditions_met = {}
    if thresholds:  
        num_conditions = len(thresholds)
        for page, conditions in detected_pages.items():
            if len(conditions) == num_conditions:
                all_conditions_met[page] = conditions
    
    return all_conditions_met

def plot_with_highlighted_pages(page_stats, highlighted_pages, output_path=None):

    fig, axes = plot_page_stream_segmentation(page_stats)
    
    for ax in axes:
        for page in highlighted_pages:
            ax.axvline(x=page, color='orange', linestyle='--', alpha=0.7)

    if highlighted_pages:
        axes[0].text(0.5, 0.95, f"Highlighted pages: {', '.join(map(str, highlighted_pages))}", 
                    transform=axes[0].transAxes, ha='center', 
                    bbox=dict(facecolor='white', alpha=0.7))
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig, axes

def analyze_comic_book_with_thresholds(file_path, output_dir=None, thresholds=None):

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    images_df, annotations_df = load_data(file_path)
    page_stats = page_stream_analysis(annotations_df)
    
    if thresholds is None:
        thresholds = {
            (1, 'area_mean'): ('>', 1.2e6),  
            (4, 'bbox_count'): ('<', 4)      
        }
    
    detected_pages = detect_pages_by_thresholds(page_stats, thresholds)
    
    highlighted_fig, _ = plot_with_highlighted_pages(
        page_stats, 
        list(detected_pages.keys()),
        output_path=f"{output_dir}/highlighted_pages.png" if output_dir else None
    )
    
    return {
        'images': images_df,
        'annotations': annotations_df,
        'page_stats': page_stats,
        'detected_pages': detected_pages
    }
    

def get_detected_imgs(results):
    
    detected_images = []
    annotations = results['annotations']
    images = results['images']
            
    
    for page, conditions in results['detected_pages'].items():

        image_id = annotations['image_id'][annotations['page_number']==page].unique()[0]
        file = np.array(images['file_name'][images['id']==image_id])[0]
        detected_images.append(file)
        
    return detected_images

def plot_images_by_ids(results, images_path):
    
    detected_images = get_detected_imgs(results)
    n_images = len(detected_images)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.array(axes).reshape(-1)  
    
    for i, file_name in enumerate(detected_images):
        try:
            image_path = os.path.join(images_path, file_name)
            image = plt.imread(image_path)
            axes[i].imshow(image)
            axes[i].set_title(f"Image: {file_name}")
            axes[i].axis('off')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error loading image {file_name}:\n{str(e)}", 
                        ha='center', va='center', fontsize=10)
            axes[i].axis('off')
            
    plt.tight_layout()
    
    return fig
