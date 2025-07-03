#!/usr/bin/env python3
"""
Fix indentation issue in live_opportunity_hunter.py
"""

def fix_indentation():
    """Fix the indentation issue in live_opportunity_hunter.py"""
    
    # Read the file with UTF-8 encoding
    try:
        with open('live_opportunity_hunter.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # Try with latin-1 encoding if UTF-8 fails
        with open('live_opportunity_hunter.py', 'r', encoding='latin-1') as f:
            lines = f.readlines()
    
    # Find the problematic section and fix it
    fixed_lines = []
    inside_problematic_section = False
    
    for i, line in enumerate(lines):
        # Check if we're at the problematic line
        if 'self.sandbox = config.get(' in line and not inside_problematic_section:
            fixed_lines.append(line)
            inside_problematic_section = True
        elif inside_problematic_section and line.strip().startswith('# Trading parameters'):
            # Start fixing indentation from here
            fixed_lines.append('        ' + line.strip() + '\n')
        elif inside_problematic_section and (line.strip().startswith('self.') or line.strip().startswith('# ')):
            # Fix indentation for these lines
            fixed_lines.append('        ' + line.strip() + '\n')
        elif inside_problematic_section and 'except FileNotFoundError:' in line:
            # End of problematic section
            inside_problematic_section = False
            fixed_lines.append(line)
        else:
            # Keep original line
            fixed_lines.append(line)
    
    # Write the fixed file with UTF-8 encoding
    with open('live_opportunity_hunter.py', 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print("✅ Fixed indentation issues in live_opportunity_hunter.py")

if __name__ == "__main__":
    fix_indentation() 