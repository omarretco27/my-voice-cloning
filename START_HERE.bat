@echo off
chcp 65001 >nul
title Riwaq Production System - GitHub Setup
cd /d "%~dp0"

echo.
echo   ==================================================
echo        Riwaq Production System - GitHub Setup
echo   ==================================================
echo.

REM ── 1) هل السكربت بجانب هذا الملف؟ ──
if not exist "%~dp0push-to-github.ps1" (
  echo   [X] لم يُعثر على push-to-github.ps1 بجانب هذا الملف.
  echo       تأكد أنك فككت ضغط الحزمة كاملة في نفس المجلد.
  echo.
  goto :end
)

REM ── 2) هل نحن داخل مجلد المشروع؟ إن لا، ابحث في مجلد فرعي واحد ──
if exist "package.json" goto :run

for /d %%D in (*) do (
  if exist "%%D\package.json" (
    echo   [i] المشروع موجود داخل المجلد الفرعي: %%D
    echo       سيتم الانتقال إليه تلقائيًا...
    cd "%%D"
    goto :run
  )
)

echo   [X] لم يُعثر على package.json في هذا المجلد ولا في أي مجلد فرعي.
echo       المجلد الحالي: %CD%
echo.
echo       تأكد أنك فككت ضغط Riwaq_System_Repo.zip بشكل صحيح بحيث
echo       تكون الملفات في جذر المجلد مباشرة:
echo       package.json . src . data . public . scripts . .github
echo.
goto :end

:run
echo   [i] مجلد العمل: %CD%
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0push-to-github.ps1"

:end
echo.
echo   ==================================================
echo   يمكنك إغلاق هذه النافذة الآن.
echo   ==================================================
pause
