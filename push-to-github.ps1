# ═══════════════════════════════════════════════════════════════════════
#  رِواق — مساعد الرفع على GitHub (Windows PowerShell)
#  الاستخدام:  powershell -ExecutionPolicy Bypass -File .\push-to-github.ps1
#  يتطلب: Git (إلزامي) · GitHub CLI (اختياري لكن يسهّل الرفع تلقائيًا)
# ═══════════════════════════════════════════════════════════════════════
$ErrorActionPreference = "Stop"

# ملاحظة: هذا السكربت يجب أن يعمل من داخل مجلد المشروع (الذي يحتوي package.json)
# إن لم يكن كذلك، سيظهر لك تنبيه واضح ويقترح الحل.

function Say($t, $c = "White") { Write-Host $t -ForegroundColor $c }

Say "╔══════════════════════════════════════════════╗" "Yellow"
Say "║   رِواق — مساعد الرفع على GitHub  (v1.0)    ║" "Yellow"
Say "╚══════════════════════════════════════════════╝" "Yellow"

# ── 1) التحقق من Git ──
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  Say "`n✘ Git غير مثبت على الجهاز." "Red"
  Say "  نزّله من: https://git-scm.com/download/win  ثم أعد تشغيل هذا السكربت." "Yellow"
  exit 1
}
Say "✔ Git موجود: $(git --version)" "Green"

# ── 2) التحقق من أننا في مجلد المشروع ──
Say "`n[i] مجلد العمل الحالي: $(Get-Location)" "Gray"
if (-not (Test-Path ".\package.json")) {
  Say "`n✘ هذا ليس مجلد المشروع (لا يوجد package.json هنا)." "Red"
  Say "  أنت الآن في: $(Get-Location)" "Yellow"
  Say ""
  Say "  الحل الأسرع: أغلق هذه النافذة واضغط مرتين على ملف:" "White"
  Say "     START_HERE.bat     (داخل مجلد المشروع)" "Cyan"
  Say ""
  Say "  أو انتقل يدويًا:   cd "E:\المسار\الصحيح\للمشروع"" "Cyan"
  Say "  ثم أعد:            powershell -ExecutionPolicy Bypass -File .\push-to-github.ps1" "Cyan"
  exit 1
}
Say "✔ مجلد المشروع صحيح." "Green"

# ── 3) هوية الالتزام ──
Say "`n── إعداد الهوية (مرة واحدة) ──" "Cyan"
$name  = Read-Host "اسمك (سيظهر في سجل المستودع)"
$email = Read-Host "إيميلك (نفس إيميل حساب GitHub)"
git config --global user.name "$name"
git config --global user.email "$email"

# ── 4) تهيئة المستودع محليًا ──
Say "`n── تجهيز المستودع المحلي ──" "Cyan"
if (-not (Test-Path ".\.git")) {
  git init -b main 2>$null
  if ($LASTEXITCODE -ne 0) { git init | Out-Null }
}
git add -A
$status = git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
  Say "لا توجد تغييرات جديدة — المستودع محدّث بالفعل." "Yellow"
} else {
  git commit -m "Riwaq Production System — v1.4.0 (locked voice + sonic v2 + golden test)" | Out-Null
  Say "✔ تم إنشاء الالتزام الأول." "Green"
}

# ── 5) الرفع ──
$gh = Get-Command gh -ErrorAction SilentlyContinue
if ($gh) {
  Say "`n── GitHub CLI متاح — الرفع تلقائي ──" "Cyan"
  $repo = Read-Host "اسم المستودع على GitHub (مثال: riwaq-system)"
  if ([string]::IsNullOrWhiteSpace($repo)) { $repo = "riwaq-system" }
  $vis = Read-Host "الخصوصية: [1] خاص Private (موصى به)  [2] عام Public  — اكتب 1 أو 2"
  $flag = if ($vis -eq "2") { "--public" } else { "--private" }

  gh auth status 2>$null
  if ($LASTEXITCODE -ne 0) {
    Say "سجّل الدخول إلى GitHub (سيفتح المتصفح):" "Yellow"
    gh auth login
  }
  gh repo create $repo $flag --source=. --push --description "Riwaq Production System — data-driven video engine"
  if ($LASTEXITCODE -eq 0) {
    $user = gh api user -q .login
    Say "`n✔ تم الرفع بنجاح!" "Green"
    Say "  المستودع: https://github.com/$user/$repo" "Green"
    Say "  الخطوة التالية: تبويب Actions → render-episode → Run workflow → اكتب GOLDEN_001" "Cyan"
    Start-Process "https://github.com/$user/$repo/actions"
  }
} else {
  Say "`n── GitHub CLI غير مثبت (اختياري) ──" "Yellow"
  Say "افعل أحد الأمرين:" "White"
  Say "  [أ] استخدم GitHub Desktop (الأسهل): https://desktop.github.com" "White"
  Say "      File → Add local repository → اختر هذا المجلد → Publish repository (Private)" "Gray"
  Say "  [ب] أو يدويًا: أنشئ مستودعًا فاضيًا على https://github.com/new (بدون README) ثم:" "White"
  Say "      git remote add origin https://github.com/USERNAME/REPO.git" "Cyan"
  Say "      git push -u origin main" "Cyan"
  Say "`n  (استبدل USERNAME و REPO باسمك واسم المستودع)" "Gray"
}
