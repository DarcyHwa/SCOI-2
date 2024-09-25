python dpp_word_diversity.py
if [ $? -ne 0 ]; then
    echo "dpp_word_diversity.py failed"
    exit 1
fi

python dpp_materials_preparation.py
if [ $? -ne 0 ]; then
    echo "dpp_materials_preparation.py failed"
    exit 1
fi

python dpp.py
if [ $? -ne 0 ]; then
    echo "dpp.py failed"
    exit 1
fi
