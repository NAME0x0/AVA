; AVA Installer Script
; ====================
; NSIS Modern UI installer for AVA - Cortex-Medulla AI Assistant
;
; Build with: makensis -DVERSION=3.2.0 ava-installer.nsi

;--------------------------------
; Includes

!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "LogicLib.nsh"
!include "WinVer.nsh"
!include "x64.nsh"

;--------------------------------
; General Configuration

!ifndef VERSION
  !define VERSION "0.0.0"
!endif

Name "AVA - Cortex-Medulla AI Assistant"
OutFile "..\dist\AVA-${VERSION}-Setup.exe"
InstallDir "$LOCALAPPDATA\AVA"
InstallDirRegKey HKCU "Software\AVA" "InstallDir"
RequestExecutionLevel user
Unicode True

; Version information
VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "AVA"
VIAddVersionKey "CompanyName" "Muhammad Afsah Mumtaz"
VIAddVersionKey "LegalCopyright" "MIT License"
VIAddVersionKey "FileDescription" "AVA Installer"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"

;--------------------------------
; Variables

Var StartMenuFolder
Var PythonInstalled
Var OllamaInstalled
Var AutoStartEnabled

;--------------------------------
; Interface Settings

!define MUI_ABORTWARNING
!define MUI_ICON "assets\installer.ico"
!define MUI_UNICON "assets\uninstaller.ico"

; Branding
!define MUI_BRANDINGTEXT "AVA v${VERSION}"

;--------------------------------
; Pages

; Welcome page
!define MUI_WELCOMEPAGE_TITLE "Welcome to AVA Setup"
!define MUI_WELCOMEPAGE_TEXT "This wizard will guide you through the installation of AVA - Cortex-Medulla AI Assistant.$\r$\n$\r$\nAVA is a research-grade AI assistant with a biomimetic dual-brain architecture.$\r$\n$\r$\nClick Next to continue."
!insertmacro MUI_PAGE_WELCOME

; License page
!insertmacro MUI_PAGE_LICENSE "license.txt"

; Components page
!define MUI_COMPONENTSPAGE_TEXT_TOP "Select the components you want to install:"
!insertmacro MUI_PAGE_COMPONENTS

; Directory page
!insertmacro MUI_PAGE_DIRECTORY

; Start Menu folder page
!define MUI_STARTMENUPAGE_REGISTRY_ROOT "HKCU"
!define MUI_STARTMENUPAGE_REGISTRY_KEY "Software\AVA"
!define MUI_STARTMENUPAGE_REGISTRY_VALUENAME "StartMenuFolder"
!define MUI_STARTMENUPAGE_DEFAULTFOLDER "AVA"
!insertmacro MUI_PAGE_STARTMENU Application $StartMenuFolder

; Custom page for options
Page custom OptionsPage OptionsPageLeave

; Installation page
!insertmacro MUI_PAGE_INSTFILES

; Finish page
!define MUI_FINISHPAGE_RUN "$INSTDIR\AVA.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Launch AVA"
!define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\README.md"
!define MUI_FINISHPAGE_SHOWREADME_TEXT "View README"
!define MUI_FINISHPAGE_LINK "Visit AVA on GitHub"
!define MUI_FINISHPAGE_LINK_LOCATION "https://github.com/NAME0x0/AVA"
!insertmacro MUI_PAGE_FINISH

; Uninstaller pages
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

;--------------------------------
; Languages

!insertmacro MUI_LANGUAGE "English"

;--------------------------------
; Custom Options Page

Function OptionsPage
  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}

  ; Title
  ${NSD_CreateLabel} 0 0 100% 20u "Startup Options"
  Pop $0
  CreateFont $1 "$(^Font)" 12 700
  SendMessage $0 ${WM_SETFONT} $1 0

  ; Auto-start checkbox
  ${NSD_CreateCheckbox} 0 30u 100% 12u "Start AVA when Windows starts"
  Pop $AutoStartEnabled
  ${NSD_SetState} $AutoStartEnabled ${BST_UNCHECKED}

  ; Info text
  ${NSD_CreateLabel} 0 50u 100% 40u "AVA will run in the system tray for quick access. You can change this setting later in the application preferences."
  Pop $0

  nsDialogs::Show
FunctionEnd

Function OptionsPageLeave
  ${NSD_GetState} $AutoStartEnabled $0
  ${If} $0 == ${BST_CHECKED}
    StrCpy $AutoStartEnabled "1"
  ${Else}
    StrCpy $AutoStartEnabled "0"
  ${EndIf}
FunctionEnd

;--------------------------------
; Installer Sections

; Required: Desktop Application (GUI)
Section "Desktop Application (Required)" SecGUI
  SectionIn RO
  SetOutPath "$INSTDIR"

  ; Copy main executable
  File /oname=AVA.exe "..\..\ui\src-tauri\target\release\ava-ui.exe"

  ; Copy resources
  SetOutPath "$INSTDIR\resources"
  File /r "..\..\ui\src-tauri\target\release\resources\*.*"

  ; Copy Python backend
  SetOutPath "$INSTDIR\backend"
  File /r "..\..\src\*.*"
  File "..\..\server.py"
  File "..\..\requirements.txt"
  File "..\..\VERSION"

  ; Copy config
  SetOutPath "$INSTDIR\config"
  File /r "..\..\config\*.*"

  ; Copy README and docs
  SetOutPath "$INSTDIR"
  File "..\..\README.md"
  File "..\..\LICENSE"

  ; Store installation directory
  WriteRegStr HKCU "Software\AVA" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "Software\AVA" "Version" "${VERSION}"

  ; Create uninstaller
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  ; Add/Remove Programs entry
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\AVA" \
                   "DisplayName" "AVA - Cortex-Medulla AI Assistant"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\AVA" \
                   "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\AVA" \
                   "DisplayIcon" "$INSTDIR\AVA.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\AVA" \
                   "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\AVA" \
                   "Publisher" "Muhammad Afsah Mumtaz"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\AVA" \
                   "URLInfoAbout" "https://github.com/NAME0x0/AVA"
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\AVA" \
                     "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\AVA" \
                     "NoRepair" 1

  ; Calculate installed size
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\AVA" \
                     "EstimatedSize" "$0"

  ; Create shortcuts
  !insertmacro MUI_STARTMENU_WRITE_BEGIN Application
    CreateDirectory "$SMPROGRAMS\$StartMenuFolder"
    CreateShortCut "$SMPROGRAMS\$StartMenuFolder\AVA.lnk" "$INSTDIR\AVA.exe"
    CreateShortCut "$SMPROGRAMS\$StartMenuFolder\Uninstall AVA.lnk" "$INSTDIR\Uninstall.exe"
  !insertmacro MUI_STARTMENU_WRITE_END

  ; Desktop shortcut
  CreateShortCut "$DESKTOP\AVA.lnk" "$INSTDIR\AVA.exe"

  ; Auto-start if enabled
  ${If} $AutoStartEnabled == "1"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Run" \
                     "AVA" '"$INSTDIR\AVA.exe" --minimized'
  ${EndIf}
SectionEnd

; Optional: Terminal Interface (TUI)
Section "Terminal Interface" SecTUI
  SetOutPath "$INSTDIR"

  ; Copy TUI files
  File "..\..\run_tui.py"

  SetOutPath "$INSTDIR\tui"
  File /r "..\..\tui\*.*"

  ; Create TUI shortcut
  !insertmacro MUI_STARTMENU_WRITE_BEGIN Application
    CreateShortCut "$SMPROGRAMS\$StartMenuFolder\AVA Terminal.lnk" \
                   "cmd.exe" '/k "$INSTDIR\run_tui.py"' \
                   "$INSTDIR\AVA.exe" 0
  !insertmacro MUI_STARTMENU_WRITE_END
SectionEnd

; Optional: CLI Tools
Section "Command Line Tools" SecCLI
  SetOutPath "$INSTDIR"

  ; Copy CLI entry point
  File "..\..\run_core.py"

  ; Add to PATH
  EnVar::SetHKCU
  EnVar::AddValue "PATH" "$INSTDIR"

  ; Create batch file for CLI
  FileOpen $0 "$INSTDIR\ava.cmd" w
  FileWrite $0 '@echo off$\r$\n'
  FileWrite $0 'python "$INSTDIR\server.py" %*$\r$\n'
  FileClose $0
SectionEnd

;--------------------------------
; Dependency Check Section

Section "-Dependencies" SecDeps
  ; Check Python
  nsExec::ExecToStack 'python --version'
  Pop $0
  Pop $1
  ${If} $0 == 0
    StrCpy $PythonInstalled "1"
    DetailPrint "Python found: $1"
  ${Else}
    StrCpy $PythonInstalled "0"
    DetailPrint "Python not found"
    MessageBox MB_YESNO|MB_ICONQUESTION \
      "Python 3.10+ is required but not found.$\r$\n$\r$\nWould you like to download Python now?" \
      IDYES download_python IDNO skip_python
    download_python:
      ExecShell "open" "https://www.python.org/downloads/"
      MessageBox MB_OK "Please install Python 3.10 or later and run this installer again."
    skip_python:
  ${EndIf}

  ; Check Ollama
  nsExec::ExecToStack 'ollama --version'
  Pop $0
  Pop $1
  ${If} $0 == 0
    StrCpy $OllamaInstalled "1"
    DetailPrint "Ollama found: $1"
  ${Else}
    StrCpy $OllamaInstalled "0"
    DetailPrint "Ollama not found"
    MessageBox MB_YESNO|MB_ICONQUESTION \
      "Ollama is required for AI functionality but not found.$\r$\n$\r$\nWould you like to download Ollama now?" \
      IDYES download_ollama IDNO skip_ollama
    download_ollama:
      ExecShell "open" "https://ollama.ai/download"
    skip_ollama:
  ${EndIf}

  ; Install Python dependencies if Python is available
  ${If} $PythonInstalled == "1"
    DetailPrint "Installing Python dependencies..."
    SetOutPath "$INSTDIR\backend"
    nsExec::ExecToLog 'python -m pip install -r "$INSTDIR\backend\requirements.txt" --quiet'
  ${EndIf}
SectionEnd

;--------------------------------
; Section Descriptions

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecGUI} \
    "Native desktop GUI with neural visualization and system tray integration. (Required)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SecTUI} \
    "Power-user terminal interface with keyboard navigation and real-time metrics."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecCLI} \
    "Command line tools for automation, scripting, and advanced usage. Adds AVA to PATH."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

;--------------------------------
; Uninstaller Section

Section "Uninstall"
  ; Remove auto-start
  DeleteRegValue HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "AVA"

  ; Remove from PATH
  EnVar::SetHKCU
  EnVar::DeleteValue "PATH" "$INSTDIR"

  ; Remove Start Menu shortcuts
  !insertmacro MUI_STARTMENU_GETFOLDER Application $StartMenuFolder
  RMDir /r "$SMPROGRAMS\$StartMenuFolder"

  ; Remove Desktop shortcut
  Delete "$DESKTOP\AVA.lnk"

  ; Remove installation directory
  RMDir /r "$INSTDIR"

  ; Remove registry entries
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\AVA"
  DeleteRegKey HKCU "Software\AVA"
SectionEnd

;--------------------------------
; Installer Functions

Function .onInit
  ; Check Windows version (require Windows 10+)
  ${IfNot} ${AtLeastWin10}
    MessageBox MB_OK|MB_ICONSTOP "AVA requires Windows 10 or later."
    Abort
  ${EndIf}

  ; Check 64-bit
  ${IfNot} ${RunningX64}
    MessageBox MB_OK|MB_ICONSTOP "AVA requires a 64-bit version of Windows."
    Abort
  ${EndIf}
FunctionEnd

Function .onInstSuccess
  ; Show completion message
  ${If} $PythonInstalled == "0"
    MessageBox MB_OK|MB_ICONINFORMATION \
      "Installation complete!$\r$\n$\r$\nNote: Python was not found. Please install Python 3.10+ to use AVA."
  ${ElseIf} $OllamaInstalled == "0"
    MessageBox MB_OK|MB_ICONINFORMATION \
      "Installation complete!$\r$\n$\r$\nNote: Ollama was not found. Please install Ollama and pull a model (e.g., 'ollama pull gemma3:4b') to use AVA."
  ${EndIf}
FunctionEnd
