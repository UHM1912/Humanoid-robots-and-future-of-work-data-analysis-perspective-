# onet_data_loader.py
import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple

class ONETDataLoader:
    """
    O*NET Database Loader and Processor
    Loads occupation data from O*NET Excel files and computes automation risk metrics
    """
    
    def __init__(self):
        self.occupations_df = None
        self.task_statements_df = None
        self.skills_df = None
        self.knowledge_df = None
        self.abilities_df = None
        self.technology_df = None
        
    def _read_csv_safe(self, path: str) -> pd.DataFrame:
        """Safely read CSV or Excel with error handling"""
        try:
            # Handle Excel files
            if path.endswith('.xlsx') or path.endswith('.xls'):
                df = pd.read_excel(path)
                return df
            # Handle CSV files
            else:
                df = pd.read_csv(path, encoding='utf-8')
                return df
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(path, encoding='latin-1')
                return df
            except Exception as e:
                st.warning(f"Could not read {path}: {e}")
                return pd.DataFrame()
        except Exception as e:
            st.warning(f"Error reading {path}: {e}")
            return pd.DataFrame()
    
    def _find_file_by_pattern(self, pattern: str, files_in_dir: List[str]) -> str:
        """Find first file matching pattern"""
        pattern_lower = pattern.lower()
        for fname in files_in_dir:
            if pattern_lower in fname.lower():
                return fname
        return None
    
    def load_onet_files(self) -> bool:
        """
        Load all O*NET files (CSV or Excel) from current directory
        """
        st.info("🔍 Searching for O*NET files (CSV or Excel)...", icon="📊")
        
        cwd_files = [f for f in os.listdir(".") if f.endswith(('.csv', '.xlsx', '.xls'))]
        
        if not cwd_files:
            st.error("❌ No CSV or Excel files found in current directory")
            return False
        
        st.info(f"Found {len(cwd_files)} files", icon="📂")
        
        # Define expected files with flexible matching
        file_patterns = {
            "occupations_df": ["occupation"],
            "task_statements_df": ["task"],
            "skills_df": ["skills"],
            "knowledge_df": ["knowledge"],
            "abilities_df": ["abilities"],
            "technology_df": ["technolog"]
        }
        
        files_loaded = 0
        for attr_name, patterns in file_patterns.items():
            found_file = None
            for pattern in patterns:
                found_file = self._find_file_by_pattern(pattern, cwd_files)
                if found_file:
                    break
            
            if found_file:
                df = self._read_csv_safe(found_file)
                if not df.empty:
                    setattr(self, attr_name, df)
                    st.success(f"✅ Loaded: {found_file} ({len(df)} rows, {len(df.columns)} cols)", icon="📈")
                    files_loaded += 1
                else:
                    st.warning(f"⚠️ File {found_file} was empty")
            else:
                st.info(f"⏭️ Skipping '{attr_name}' - file not found (optional)")
        
        if files_loaded == 0:
            st.error("❌ No valid O*NET files loaded")
            return False
        
        st.success(f"✅ O*NET Loader: {files_loaded} files loaded successfully", icon="🎯")
        return True
    
    def compute_automation_risk_score(self) -> pd.DataFrame:
        """
        Compute automation risk score based on O*NET occupational data
        Uses task and skill data to assess automation vulnerability
        """
        if self.occupations_df is None or self.occupations_df.empty:
            st.error("❌ Occupations data not loaded")
            return pd.DataFrame()
        
        st.info("🧮 Computing automation risk scores from O*NET data...", icon="⚙️")
        
        risk_data = []
        occupations_df = self.occupations_df.copy()
        
        # Display comprehensive debugging information
        st.write("### 🔍 Detailed O*NET Data Analysis")
        st.write(f"**Total Occupations:** {len(occupations_df)}")
        st.write(f"**Total Columns:** {len(occupations_df.columns)}")
        st.write(f"**Column Names:** {list(occupations_df.columns)}")
        
        # Show first few rows
        with st.expander("📊 Show first 5 rows of data"):
            st.dataframe(occupations_df.head())
        
        # Show numeric columns
        numeric_cols = occupations_df.select_dtypes(include=[np.number]).columns.tolist()
        st.write(f"**Numeric Columns Found:** {len(numeric_cols)}")
        if numeric_cols:
            st.write(f"Columns: {numeric_cols}")
            with st.expander("📈 Show numeric data sample"):
                st.dataframe(occupations_df[numeric_cols].head(10))
        else:
            st.warning("⚠️ No numeric columns found! Checking for numeric strings...")
            
            # Try to find columns with numeric-looking data
            for col in occupations_df.columns:
                sample_val = str(occupations_df[col].iloc[0])
                try:
                    float(sample_val)
                    st.info(f"Found numeric string column: {col}")
                except:
                    pass
        
        # Find occupation identifier columns
        occ_code_col = self._find_column(occupations_df, ["O*NET-SOC Code", "O*NET-SOC", "Code", "SOC Code", "code"])
        occ_title_col = self._find_column(occupations_df, ["Title", "Occupation", "Job Title", "title", "job title"])
        
        if not occ_code_col:
            occ_code_col = occupations_df.columns[0]
        if not occ_title_col:
            occ_title_col = occupations_df.columns[1] if len(occupations_df.columns) > 1 else occupations_df.columns[0]
        
        st.success(f"✅ Using Code Column: **{occ_code_col}** | Title Column: **{occ_title_col}**")
        
        try:
            for idx, row in occupations_df.iterrows():
                occ_code = str(row.get(occ_code_col, f"OCC_{idx}")).strip()
                occ_title = str(row.get(occ_title_col, f"Occupation {idx}")).strip()
                
                # Get all numeric columns and values
                numeric_values = []
                
                if numeric_cols:
                    # Use actual numeric columns
                    for col in numeric_cols:
                        try:
                            val = float(row.get(col, 0.5))
                            numeric_values.append(val)
                        except:
                            pass
                else:
                    # Try to convert string columns to numeric
                    for col in occupations_df.columns[2:]:  # Skip first 2 (code, title)
                        try:
                            val = float(str(row.get(col, "0.5")).replace(",", ""))
                            numeric_values.append(val)
                        except:
                            pass
                
                # Extract or compute components
                if len(numeric_values) >= 4:
                    routine_intensity = self._normalize_value(numeric_values[0])
                    manual_intensity = self._normalize_value(numeric_values[1])
                    cognitive_complexity = self._normalize_value(numeric_values[2])
                    human_interaction = self._normalize_value(numeric_values[3])
                elif len(numeric_values) > 0:
                    # Distribute available values
                    routine_intensity = self._normalize_value(numeric_values[0])
                    manual_intensity = self._normalize_value(numeric_values[0])
                    cognitive_complexity = self._normalize_value(numeric_values[-1]) if len(numeric_values) > 1 else 0.5
                    human_interaction = self._normalize_value(numeric_values[-1]) if len(numeric_values) > 1 else 0.5
                else:
                    routine_intensity = 0.5
                    manual_intensity = 0.5
                    cognitive_complexity = 0.5
                    human_interaction = 0.5
                
                # Compute automation risk
                automation_risk = (routine_intensity + manual_intensity) * 0.4 + \
                                 (1 - cognitive_complexity) * 0.3 + \
                                 (1 - human_interaction) * 0.3
                automation_risk = np.clip(automation_risk, 0, 1)
                
                risk_data.append({
                    "O*NET Code": occ_code,
                    "Occupation": occ_title,
                    "Routine Intensity": routine_intensity,
                    "Manual Intensity": manual_intensity,
                    "Cognitive Complexity": cognitive_complexity,
                    "Human Interaction": human_interaction,
                    "Automation Risk Score": automation_risk,
                    "Risk Level": self._categorize_risk(automation_risk)
                })
        
        except Exception as e:
            st.error(f"❌ Error computing automation risk: {e}")
            import traceback
            st.error(traceback.format_exc())
            return pd.DataFrame()
        
        risk_df = pd.DataFrame(risk_data)
        st.success(f"✅ Computed automation risk for {len(risk_df)} occupations", icon="📊")
        
        # Show distribution
        high_risk = len(risk_df[risk_df["Automation Risk Score"] > 0.67])
        med_risk = len(risk_df[(risk_df["Automation Risk Score"] >= 0.33) & (risk_df["Automation Risk Score"] <= 0.67)])
        low_risk = len(risk_df[risk_df["Automation Risk Score"] < 0.33])
        st.info(f"🟢 Low: {low_risk} | 🟡 Medium: {med_risk} | 🔴 High: {high_risk}")
        
        return risk_df
    
    def compute_skill_analysis(self) -> pd.DataFrame:
        """
        Analyze skills data directly from O*NET
        """
        if self.skills_df is None or self.skills_df.empty:
            st.warning("⚠️ Skills data not available")
            return pd.DataFrame()
        
        try:
            skill_data = self.skills_df.copy()
            
            st.info(f"📋 Skills columns: {list(skill_data.columns[:5])}")
            
            # Find skill and value columns
            skill_col = self._find_column(skill_data, ["Title", "Skill", "Name", "skill name"])
            value_col = self._find_column(skill_data, ["Data Value", "Value", "Level", "Importance", "data value"])
            
            if not skill_col:
                skill_col = skill_data.columns[0]
            if not value_col:
                numeric_cols = skill_data.select_dtypes(include=[np.number]).columns
                value_col = numeric_cols[0] if len(numeric_cols) > 0 else None
            
            if not value_col:
                st.warning("⚠️ Could not find numeric value column in skills data")
                return pd.DataFrame()
            
            result = skill_data.copy()
            result["Skill"] = result[skill_col].astype(str)
            result["Level"] = pd.to_numeric(result[value_col], errors="coerce").fillna(0)
            
            # Normalize
            max_level = result["Level"].max()
            result["Level Normalized"] = result["Level"] / max_level if max_level > 0 else result["Level"]
            
            result["Transferability"] = result["Level Normalized"].apply(
                lambda x: "High" if x > 0.7 else "Medium" if x > 0.4 else "Low"
            )
            
            return result[["Skill", "Level", "Level Normalized", "Transferability"]].drop_duplicates()
        
        except Exception as e:
            st.error(f"❌ Error analyzing skills: {e}")
            return pd.DataFrame()
    
    def compute_technology_analysis(self) -> pd.DataFrame:
        """
        Analyze technology data directly from O*NET
        """
        if self.technology_df is None or self.technology_df.empty:
            st.warning("⚠️ Technology data not available")
            return pd.DataFrame()
        
        try:
            tech_data = self.technology_df.copy()
            
            st.info(f"📋 Technology columns: {list(tech_data.columns[:5])}")
            
            # Find relevant columns
            tech_col = self._find_column(tech_data, ["Technology Example", "Technology", "Name", "Title", "tech"])
            value_col = self._find_column(tech_data, ["Data Value", "Value", "Level", "Importance"])
            
            if not tech_col:
                tech_col = tech_data.columns[0]
            if not value_col:
                numeric_cols = tech_data.select_dtypes(include=[np.number]).columns
                value_col = numeric_cols[0] if len(numeric_cols) > 0 else None
            
            if not value_col:
                st.warning("⚠️ Could not find numeric value column in technology data")
                return pd.DataFrame()
            
            result = tech_data.copy()
            result["Technology"] = result[tech_col].astype(str)
            result["Adoption Level"] = pd.to_numeric(result[value_col], errors="coerce").fillna(0)
            
            # Normalize
            max_adoption = result["Adoption Level"].max()
            result["Adoption Normalized"] = result["Adoption Level"] / max_adoption if max_adoption > 0 else result["Adoption Level"]
            
            return result[["Technology", "Adoption Level", "Adoption Normalized"]].drop_duplicates()
        
        except Exception as e:
            st.error(f"❌ Error analyzing technology: {e}")
            return pd.DataFrame()
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> str:
        """Find column name from possible variations"""
        df_cols_lower = [col.lower() for col in df.columns]
        for name in possible_names:
            name_lower = name.lower()
            for i, col in enumerate(df_cols_lower):
                if name_lower in col or col in name_lower:
                    return df.columns[i]
        return None
    
    def _normalize_value(self, value) -> float:
        """Normalize value to 0-1 range"""
        if pd.isna(value):
            return 0.5
        
        try:
            value = float(value)
            if value > 100:
                return np.clip(value / 100, 0, 1)
            return np.clip(value, 0, 1)
        except:
            return 0.5
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize automation risk"""
        if risk_score < 0.33:
            return "🟢 Low Risk"
        elif risk_score < 0.67:
            return "🟡 Medium Risk"
        else:
            return "🔴 High Risk"


def load_onet_analysis() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main function to load and analyze O*NET data
    Returns: (automation_risk_df, skills_df, technology_df)
    """
    loader = ONETDataLoader()
    
    # Load files
    if not loader.load_onet_files():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Compute metrics
    risk_df = loader.compute_automation_risk_score()
    skills_df = loader.compute_skill_analysis()
    tech_df = loader.compute_technology_analysis()
    
    return risk_df, skills_df, tech_df