#!/bin/bash
# Master Script for AGI GitHub Release
# Run all applications and save logs

echo "==============================================="
echo "🚀 AGI GitHub Release - Application Runner"
echo "==============================================="

# Set up environment
export PYTHONPATH=.
LOG_DIR="logs"
mkdir -p $LOG_DIR

run_app() {
    APP_NAME=$1
    COMMAND=$2
    LOG_FILE="${LOG_DIR}/${APP_NAME}.log"
    
    echo "-----------------------------------------------"
    echo "🔄 Running: ${APP_NAME}"
    echo "Command: ${COMMAND}"
    echo "Log: ${LOG_FILE}"
    
    eval "${COMMAND}" 2>&1 | tee -a "${LOG_FILE}"
    
    if [ $? -eq 0 ]; then
        echo "✅ ${APP_NAME} completed successfully"
    else
        echo "❌ ${APP_NAME} encountered an error"
    fi
    echo ""
}

# Generate Demo License
LICENSE_KEY="540e4a27d374b9cd58add850949aeed4595ee582570252db538bdb3776d7aa98cd7614c533640914d1df5e03462ff9247b3ff385bff7ebd5b04de66b09c1c231"
export MRZKELP_LICENSE_KEY="$LICENSE_KEY"
export MRZKELP_CLIENT_ID="demo@example.com"
export MRZKELP_SECRET="AGIToolkitMaster"

echo "-----------------------------------------------"
echo "🔑 License Key Generated and Set"
echo "-----------------------------------------------"

# Create sample files if needed
if [ ! -f "test_sample.txt" ]; then
    echo "Creating sample text file for testing..."
    cat > test_sample.txt << EOF
Artificial General Intelligence (AGI) refers to a type of artificial intelligence that has the ability to understand, learn, and apply knowledge across a wide range of tasks, similar to human intelligence. Unlike narrow AI, which is designed to perform specific tasks, AGI would have a more flexible, general capability to solve various problems.

The development of AGI raises important questions about safety, ethics, and control. Researchers work on ensuring that AGI systems align with human values and goals. This includes problems like interpretability (understanding why an AI makes certain decisions), robustness (ensuring AI performs well even in unusual circumstances), and value alignment (ensuring AI's goals match human intentions).
EOF
fi

echo "==============================================="
echo "🏦 1. Running Banking App"
echo "==============================================="

# Banking App - Normal Mode
run_app "banking" "python3 real_world_apps/banking/app.py ingest --account-id demo1 --tx-id tx001 --amount 100 --description 'Demo transaction'"

# Banking App - Show Transactions
run_app "banking_show" "python3 real_world_apps/banking/app.py show-transactions --account-id demo1"

# Banking App - Corporate Mode with License
run_app "banking_enterprise" "SIMULATE_CORPORATE=true python3 real_world_apps/banking/app.py analyze-account --account-id demo1"

echo "==============================================="
echo "🪖 2. Running Military Logistics"
echo "==============================================="

run_app "military_logistics" "python3 real_world_apps/military_logistics/app.py --demo"

echo "==============================================="
echo "📝 3. Running Content Summarizer"
echo "==============================================="

run_app "content_summarizer" "python3 real_world_apps/content_summarizer/app.py --file test_sample.txt --length short"

echo "==============================================="
echo "📄 4. Running Document Assistant"
echo "==============================================="

run_app "document_assistant" "python3 real_world_apps/document_assistant/app.py --file test_sample.txt"

echo "==============================================="
echo "🎓 5. Running Learning Platform"
echo "==============================================="

run_app "learning_platform" "python3 real_world_apps/learning_platform/app.py --demo"

echo "==============================================="
echo "😀 6. Running Sentiment Dashboard"
echo "==============================================="

run_app "sentiment_dashboard" "python3 real_world_apps/sentiment_dashboard/app.py --text 'I really love the AGI Toolkit! It makes building AI applications so much easier.'"

echo "==============================================="
echo "🌐 7. Running Translation Service"
echo "==============================================="

run_app "translation_service" "python3 real_world_apps/translation_service/app.py --text 'Hello, how are you today?' --source en --target es"

echo "==============================================="
echo "🤖 8. Running Virtual Assistant"
echo "==============================================="

run_app "virtual_assistant" "python3 real_world_apps/virtual_assistant/app.py --demo"

echo "==============================================="
echo "📊 SUMMARY"
echo "==============================================="
echo "All applications have been run and logs saved to ${LOG_DIR} directory."
echo "View the summary report at: ${LOG_DIR}/summary_report.md"
echo ""
echo "To generate a new license:"
echo "python3 /home/umesh/license_manager.py generate --client-id \"client@company.com\" --company \"Company Name\" --type corporate"
echo ""
echo "==============================================="
