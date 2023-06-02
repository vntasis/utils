#!/bin/bash
set -e

# Function to print the help page
print_help() {
    cat << EOF
Project Initialization Script
This script creates a new project directory with a custom structure and initializes it as a git project.

Requirements:
- Git
- Basic GNU utilities (mkdir, cd, pwd, echo, ...)

Usage: ./project-init.sh [OPTIONS]
Options:
  -n, --name NAME             Specify the project name (required)
  -d, --description TEXT      Specify the project description. Default: "A new project about NAME"
  -r, --remote REPO_URL       Specify the remote repository URL
  -s, --structure DIRS        Specify the project subdirectory structure (comma-separated). Default: "plots,scripts,data,reports,templates"
  -i, --ignore DIRS           Specify directories to be ignored by git (comma-separated). Default: "plots,data,templates"
  -h, --help                  Display this help page

Examples:
  ./project-init.sh -n myproject -d "My awesome project" -r https://gitlab.com/myusername/myproject.git
  ./project-init.sh --name myproject --structure scripts,docs,data --ignore docs
EOF
}

# Function to check if a value is empty or starts with a hyphen
check_value() {
    local value="$1"
    local name="$2"

    if [[ -z "$value" || "$value" == -* ]]; then
        echo "Error: $name not provided."
        exit 1
    fi
}

# Function to parse command-line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
            -n|--name)
                check_value "$2" "Project name"
                project_name="$2"
                shift 2
                ;;
            -d|--description)
                check_value "$2" "Project description"
                project_description="$2"
                shift 2
                ;;
            -r|--remote)
                check_value "$2" "Remote repository URL"
                remote_repo="$2"
                shift 2
                ;;
            -s|--structure)
                check_value "$2" "Project structure"
                structure="$2"
                shift 2
                ;;
            -i|--ignore)
                check_value "$2" "Ignored directories"
                ignore_dirs="$2"
                shift 2
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            *)
                echo "Invalid option: $key"
                print_help
                exit 1
                ;;
        esac
    done

    # Check if project name is empty
    if [ -z "$project_name" ]; then
        echo "Project name not specified. Use -n or --name option."
        exit 1
    fi
}

# Function for the main script logic
create_project() {
    # Create the project directory
    mkdir "$project_name"
    cd "$project_name"
    project_dir=$(pwd)

    # Initialize the directory as a git repository
    git init

    # Set the project description
    if [ -z "$project_description" ]; then
        project_description="A new project about $project_name"
    fi
    echo "# Project Description" > readme.md
    echo "$project_description" >> readme.md

    # Create custom subdirectories
    IFS=',' read -ra dirs <<< "$structure"
    for dir in "${dirs[@]}"; do
        mkdir "$dir"
    done

    # Create .gitignore file
    IFS=',' read -ra ignore_arr <<< "$ignore_dirs"
    for ignore_dir in "${ignore_arr[@]}"; do
        echo "$ignore_dir/" >> .gitignore
    done

    # Add remote repository if provided
    if [ -n "$remote_repo" ]; then
        git remote add origin "$remote_repo"
    fi

    # Generate headers for additional sections in README file
    cat << EOF >> readme.md

## Data Description

## Analysis Description

## Scripts Description

## Results Summary

## Conclusion and Future Work
EOF

    # Navigate out of the created directory
    cd ..

    # Print the path of the created project directory
    echo "Project directory created at: $project_dir"
}

# Main script
main() {
    # Default values
    project_name=""
    project_description=""
    remote_repo=""
    structure="plots,scripts,data,reports,templates"
    ignore_dirs="plots,data,templates"

    # Parse command-line arguments
    parse_arguments "$@"

    # Create the project
    create_project
}

# Execute the main script
main "$@"
