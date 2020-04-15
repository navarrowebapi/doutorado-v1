$FILES = ls *.pdf
foreach ($f in $FILES) {
    C:\Users\fpnav\Downloads\xpdf-tools-win-4.02\bin64\pdftotext.exe -enc Latin1 $f
}