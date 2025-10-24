#!/usr/bin/env python3
"""
Script to create Course Handout Excel file for 21CSE558T Deep Neural Network Architectures
This script creates a multi-sheet Excel workbook with proper formatting
"""

import pandas as pd
from datetime import datetime
import os

def create_course_handout_excel():
    """Create the Course Handout Excel file with multiple sheets"""
    
    # Create Excel writer object
    filename = 'Course_Handout_21CSE558T.xlsx'
    writer = pd.ExcelWriter(filename, engine='openpyxl')
    
    # Sheet 1: Course Overview
    course_overview = {
        'Field': [
            'Course Code', 'Course Name', 'Course Category', 'Credits (L-T-P-C)',
            'Pre-requisite Courses', 'Co-requisite Courses', 'Progressive Courses',
            'Course Offering Department', 'Academic Year', 'Semester'
        ],
        'Details': [
            '21CSE558T', 'DEEP NEURAL NETWORK ARCHITECTURES', 'Professional Elective', '2-1-0-3',
            'Nil', 'Nil', 'Nil',
            'School of Computing', '2025-26', 'VII'
        ]
    }
    df_overview = pd.DataFrame(course_overview)
    df_overview.to_excel(writer, sheet_name='Course Overview', index=False)
    
    # Sheet 2: Learning Outcomes
    learning_outcomes = {
        'Type': ['CLR-1', 'CLR-2', 'CLR-3', 'CLR-4', 'CLR-5', '', 'CO-1', 'CO-2', 'CO-3', 'CO-4', 'CO-5'],
        'Description': [
            'understand the fundamental concepts and basic tools of deep neural networks',
            'recognize and appreciate the functionalities of various layers in deep neural networks',
            'explore the application of deep neural networks in image processing',
            'comprehend convolutional neural networks and its layer wise functionality',
            'get familiar with transfer learning techniques',
            '',
            'create a simple deep neural network and explain its functions',
            'build neural networks with multiple layers with appropriate activations',
            'apply deep neural networks in image processing problems',
            'implement convolutional neural networks',
            'determine the application of appropriate transfer learning techniques'
        ]
    }
    df_outcomes = pd.DataFrame(learning_outcomes)
    df_outcomes.to_excel(writer, sheet_name='Learning Outcomes', index=False)
    
    # Sheet 3: CO-PO Mapping
    co_po_mapping = {
        'Course Outcomes': ['CO-1', 'CO-2', 'CO-3', 'CO-4', 'CO-5'],
        'PO-1': [3, 1, 0, 2, 1],
        'PO-2': [0, 2, 2, 0, 3],
        'PO-3': [1, 0, 1, 1, 0],
        'PO-4': [0, 0, 0, 0, 0],
        'PO-5': [0, 0, 0, 0, 0]
    }
    df_mapping = pd.DataFrame(co_po_mapping)
    df_mapping.to_excel(writer, sheet_name='CO-PO Mapping', index=False)
    
    # Sheet 4: Module Content
    modules_data = {
        'Module': ['Module-1', 'Module-2', 'Module-3', 'Module-4', 'Module-5'],
        'Title': [
            'Introduction to Deep Learning',
            'Optimization and Regularization',
            'Image Processing and Deep Neural Networks',
            'Convolutional Neural Networks and Transfer Learning',
            'Object Localization and Detection Models'
        ],
        'Hours': [9, 9, 9, 9, 9],
        'Key Topics': [
            'Fundamentals, Biological Neurons, Perceptron, TensorFlow Basics, Activation Functions',
            'Gradient Descent, Regularization, Normalization, Optimization Algorithms',
            'Image Processing, Feature Extraction, Computer Vision Applications',
            'CNN Architecture, Transfer Learning, Pre-trained Models',
            'Object Detection, YOLO, R-CNN, Evaluation Metrics'
        ],
        'Tutorial Tasks': [
            'T1: TensorFlow Environment, T2: Tensors, T3: Basic Operations',
            'T4: Basic Neural Network, T5: Keras, T6: Gradient Descent',
            'T7: Image Processing, T8: Segmentation, T9: Feature Extraction',
            'T10: CNN Classification, T11: Data Augmentation, T12: LSTM',
            'T13: Pre-trained Models, T14: Fine-tuning, T15: Object Detection'
        ]
    }
    df_modules = pd.DataFrame(modules_data)
    df_modules.to_excel(writer, sheet_name='Module Content', index=False)
    
    # Sheet 5: Assessment Plan
    assessment_data = {
        'Component': [
            'Formative Assessment I (Quiz)',
            'Formative Assessment II (Written)',
            'Formative Assessment III (Written)',
            'Formative Assessment IV (Tutorials)',
            'Lifelong Learning (Project)',
            'Final Examination'
        ],
        'Marks': [5, 15, 15, 15, 10, 40],
        'Weightage (%)': [8.33, 25, 25, 25, 16.67, 40],
        'Tentative Date': [
            '29-Aug-2025',
            '19-Sep-2025',
            '31-Oct-2025',
            'Continuous',
            'End of Module 3',
            'Nov 2025'
        ],
        'Duration': ['50 min', '100 min', '100 min', 'Weekly', 'Presentation', '3 hours'],
        'Coverage': [
            'Module 1 (Partial)',
            'Modules 1 & 2',
            'Modules 3 & 4',
            'All Modules',
            'Research Component',
            'All Modules'
        ]
    }
    df_assessment = pd.DataFrame(assessment_data)
    df_assessment.to_excel(writer, sheet_name='Assessment Plan', index=False)
    
    # Sheet 6: Weekly Schedule
    weekly_schedule = []
    week_topics = [
        # Week 1
        ['Fundamentals of Deep Learning', 'Biological Neurons', 'Perceptron Model'],
        # Week 2
        ['Multilayer Perceptron', 'TensorFlow Basics', 'Data Structures'],
        # Week 3
        ['Activation Functions', 'Neural Network Layers', 'Mathematical Models'],
        # Week 4
        ['Gradient Descent', 'SGD and Mini-batch', 'Vanishing Gradients'],
        # Week 5
        ['Overfitting/Underfitting', 'Hyperparameter Tuning', 'Regularization'],
        # Week 6
        ['Early Stopping', 'Normalization', 'Advanced Optimization'],
        # Week 7
        ['Image Processing Fundamentals', 'Image Enhancement', 'Edge Detection'],
        # Week 8
        ['Image Segmentation', 'ROI Processing', 'Feature Extraction'],
        # Week 9
        ['Unstructured Data', 'Image Classification', 'CV Applications'],
        # Week 10
        ['CNN Motivation', 'Convolution Operations', 'CNN Architecture'],
        # Week 11
        ['Pooling Layers', 'Fully Connected', 'CNN Regularization'],
        # Week 12
        ['Stride Convolutions', 'Transfer Learning', 'Pre-trained Models'],
        # Week 13
        ['Object Localization', 'YOLO Architecture', 'SSD Framework'],
        # Week 14
        ['R-CNN Family', 'Fast/Faster R-CNN', 'Region Proposals'],
        # Week 15
        ['IoU Metrics', 'mAP Evaluation', 'Non-maximal Suppression']
    ]
    
    tutorials = [
        'T1: TensorFlow Environment', 'T2: Working with Tensors', 'T3: Basic Operations',
        'T4: Basic Neural Network', 'T5: Keras Implementation', 'T6: Gradient Descent',
        'T7: Image Processing', 'T8: Image Segmentation', 'T9: Feature Extraction',
        'T10: CNN Classification', 'T11: Data Augmentation', 'T12: LSTM Model',
        'T13: Pre-trained Models', 'T14: Transfer Learning', 'T15: Object Detection'
    ]
    
    for week in range(1, 16):
        module = 1 if week <= 3 else 2 if week <= 6 else 3 if week <= 9 else 4 if week <= 12 else 5
        for session in range(3):
            if week <= 15 and session < len(week_topics[week-1]):
                weekly_schedule.append({
                    'Week': week,
                    'Session': session + 1,
                    'Module': f'Module-{module}',
                    'Topic': week_topics[week-1][session],
                    'Hours': 1,
                    'Method': 'Lecture' if session != 2 else 'Lab',
                    'Tutorial': tutorials[week-1] if session == 2 else '',
                    'Assessment': 'Continuous'
                })
    
    df_schedule = pd.DataFrame(weekly_schedule)
    df_schedule.to_excel(writer, sheet_name='Weekly Schedule', index=False)
    
    # Sheet 7: Resources
    resources_data = {
        'Category': ['Hardware', 'Hardware', 'Hardware', 'Software', 'Software', 'Software', 'Software', 'Books', 'Books', 'Books'],
        'Item': [
            'GPU-enabled Systems',
            'High RAM (16GB+)',
            'Storage (100GB+)',
            'Python 3.7+',
            'TensorFlow 2.x',
            'Keras',
            'OpenCV 4.x',
            'Deep Learning with Python (Chollet)',
            'Deep Learning (Goodfellow et al.)',
            'Machine Learning (Murphy)'
        ],
        'Description': [
            'NVIDIA GPU support for deep learning',
            'Sufficient memory for large datasets',
            'Space for datasets and models',
            'Primary programming language',
            'Deep learning framework',
            'High-level neural networks API',
            'Computer vision library',
            'Primary textbook',
            'Comprehensive reference',
            'Mathematical foundations'
        ],
        'Status': ['Required', 'Required', 'Required', 'Required', 'Required', 'Required', 'Required', 'Required', 'Reference', 'Reference']
    }
    df_resources = pd.DataFrame(resources_data)
    df_resources.to_excel(writer, sheet_name='Resources', index=False)
    
    # Save the Excel file
    writer.close()
    print(f"Course Handout Excel file created successfully: {filename}")
    print(f"File location: {os.path.abspath(filename)}")
    
    return filename

if __name__ == "__main__":
    create_course_handout_excel()