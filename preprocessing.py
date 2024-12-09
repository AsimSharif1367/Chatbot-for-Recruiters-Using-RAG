import pandas as pd
import numpy as np
import re
import os
from nltk.tokenize import RegexpTokenizer
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ResumeDatasetLoader:
    def __init__(self, dataset_name: str = "UpdatedResumeDataSet.csv"):
        self.data_path = dataset_name
        self.logger = logging.getLogger('ResumeDatasetLoader')
        
    def load_dataset(self) -> pd.DataFrame:
        try:
            self.logger.info(f"Looking for dataset at: {os.path.abspath(self.data_path)}")
            
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Dataset not found at {self.data_path}")
            
            df = pd.read_csv(self.data_path)
            self.logger.info(f"Successfully loaded {len(df)} resumes")
            self.logger.info(f"Dataset shape: {df.shape}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise

class ResumePreprocessor:
    def __init__(self):
        self.logger = logging.getLogger('ResumePreprocessor')
        self.word_tokenizer = RegexpTokenizer(r'\w+')
        
        # Define section headers
        self.section_headers = {
            'education': [
                r'\b(?:education|academic|qualification)s?\b',
                r'\beducational (?:background|qualification)s?\b',
                r'\bacademic (?:background|qualification)s?\b',
                r'\buniversity|college|school\b',
                r'\bdegree\b'
            ],
            'experience': [
                r'\b(?:work |professional )?experience\b',
                r'\bemployment(?: history)?\b',
                r'\bwork history\b',
                r'\bcareer(?: history| summary)?\b',
                r'\bjob history\b',
                r'\bprofessional background\b'
            ],
            'skills': [
                r'\btechnical skills\b',
                r'\bskills(?: set)?\b',
                r'\bcore competencies\b',
                r'\bkey skills\b',
                r'\bproficiencies\b',
                r'\btechnical expertise\b',
                r'\bskill highlights\b'
            ],
            'projects': [
                r'\bprojects?\b',
                r'\bproject experience\b',
                r'\bkey projects\b',
                r'\bmajor projects\b',
                r'\bacademic projects\b',
                r'\bproject details\b'
            ],
            'summary': [
                r'\b(?:professional )?summary\b',
                r'\bprofile\b',
                r'\bcareer objective\b',
                r'\bobjective\b',
                r'\babout(?: me)?\b',
                r'\bintroduction\b'
            ]
        }
        
        # Define technical skills categories
        self.technical_skills = {
            'programming_languages': {
                'python', 'java', 'javascript', 'cpp', 'c++', 'ruby', 'php', 'scala',
                'html', 'css', 'sql', 'nosql', 'rust', 'julia', 'matlab', 'r',
                'swift', 'kotlin', 'typescript', 'go', 'perl', 'shell', 'bash'
            },
            'frameworks': {
                'django', 'flask', 'spring', 'react', 'angular', 'vue', 
                'tensorflow', 'pytorch', 'keras', 'scikit', 'pandas', 'numpy',
                'scipy', 'matplotlib', 'seaborn', 'plotly', 'fastapi', 'express',
                'bootstrap', 'jquery', 'node.js', 'laravel', 'asp.net'
            },
            'databases': {
                'mysql', 'postgresql', 'mongodb', 'cassandra', 'oracle',
                'sqlite', 'redis', 'elasticsearch', 'dynamodb', 'mariadb',
                'neo4j', 'couchdb', 'hbase', 'sql server'
            },
            'tools_and_platforms': {
                'git', 'docker', 'kubernetes', 'jenkins', 'jira', 'aws',
                'azure', 'gcp', 'hadoop', 'spark', 'tableau', 'power bi',
                'linux', 'unix', 'vscode', 'intellij', 'postman', 'nginx',
                'maven', 'gradle', 'jupyter', 'airflow', 'kafka'
            },
            'machine_learning': {
                'machine learning', 'deep learning', 'nlp', 'computer vision',
                'neural networks', 'clustering', 'classification', 'regression',
                'supervised learning', 'unsupervised learning', 'reinforcement learning',
                'time series', 'natural language processing', 'ai'
            }
        }

    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        for section, patterns in self.section_headers.items():
            for pattern in patterns:
                text = re.sub(
                    f'(^|\n)({pattern})(:|\.|\n|$)',
                    f'\n=={section.upper()}==\n',
                    text,
                    flags=re.IGNORECASE | re.MULTILINE
                )
        
        # Clean personal information
        text = re.sub(r'\S+@\S+', 'EMAIL', text)
        text = re.sub(r'\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}', 'PHONE', text)
        text = re.sub(r'http\S+|www.\S+', 'URL', text)
        
        # Clean whitespace
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(line for line in lines if line)
        
        return text

    def identify_sections(self, text: str) -> Dict[str, str]:
        sections = {k: '' for k in self.section_headers.keys()}
        chunks = re.split(r'==(\w+)==', text)
        
        if len(chunks) > 1:
            current_section = None
            for chunk in chunks:
                chunk = chunk.strip()
                if chunk in sections.keys():
                    current_section = chunk.lower()
                elif current_section and chunk:
                    sections[current_section] += chunk + '\n'
        else:
            lines = text.split('\n')
            current_section = None
            buffer = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                found_section = None
                for section, patterns in self.section_headers.items():
                    if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns):
                        found_section = section
                        break
                
                if found_section:
                    if current_section and buffer:
                        sections[current_section] = '\n'.join(buffer)
                    current_section = found_section
                    buffer = []
                elif current_section:
                    buffer.append(line)
            
            if current_section and buffer:
                sections[current_section] = '\n'.join(buffer)
        
        return sections

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        text_lower = text.lower()
        found_skills = {}
        
        for category, skills in self.technical_skills.items():
            found = set()
            words = set(self.word_tokenizer.tokenize(text_lower))
            found.update(skill for skill in skills if skill in words)
            
            for skill in skills:
                if ' ' in skill and skill in text_lower:
                    found.add(skill)
                elif '-' in skill and skill.replace('-', ' ') in text_lower:
                    found.add(skill)
            
            found_skills[category] = sorted(found)
        
        return found_skills

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Starting resume preprocessing...")
        
        processed_data = []
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                self.logger.info(f"Processing resume {idx}")
            
            cleaned_text = self.clean_text(row['Resume'])
            sections = self.identify_sections(cleaned_text)
            skills = self.extract_skills(cleaned_text)
            
            processed_data.append({
                'category': row['Category'],
                'sections': sections,
                'skills': skills,
                'original_text': row['Resume'],
                'cleaned_text': cleaned_text
            })
        
        self.logger.info("Resume preprocessing completed!")
        return pd.DataFrame(processed_data)

def test_preprocessing():
    """Test function to verify preprocessing functionality"""
    try:
        # Test dataset loading
        loader = ResumeDatasetLoader()
        df = loader.load_dataset()
        print(f"Successfully loaded dataset with {len(df)} resumes")
        
        # Test preprocessing
        preprocessor = ResumePreprocessor()
        processed_df = preprocessor.process_dataframe(df)
        print("\nPreprocessing completed successfully!")
        print(f"Processed {len(processed_df)} resumes")
        
        # Display sample results
        print("\nSample processed resume:")
        sample = processed_df.iloc[0]
        print(f"Category: {sample['category']}")
        print("\nExtracted Skills:")
        for category, skills in sample['skills'].items():
            if skills:
                print(f"{category}: {', '.join(skills)}")
        
        return True
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    # Run tests when file is executed directly
    test_preprocessing()