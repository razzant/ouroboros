; The per-user Windows installer of Ouroboros. scripts\pack_windows_setup.ps1 compiles
; it with the pinned Inno Setup (inno_setup_pin.json) from an existing collected
; payload, the Ouroboros folder the portable ZIP holds, and installs exactly that folder.
;
; It installs without elevation into %LOCALAPPDATA%\Programs\Ouroboros and adds one Start
; menu shortcut and an ordinary uninstaller. The desktop app takes the shortcut's taskbar
; identity only there and only once this Setup's HKCU registration proves the copy is
; Setup's (launcher_windows_runtime.setup_install_root mirrors IsOwnedInstallDir).
; Upgrades install in place over the same AppId, appending to Inno Setup's uninstall
; log without wiping the previous installation up front: cancellation is allowed through
; the Welcome, Ready and Preparing pages (and busy-file retry prompts) before any file is
; touched; once file extraction begins, AllowCancelDuringInstall=no disables normal
; cancellation so a cancelled upgrade cannot leave mixed versions (an I/O error or forced
; termination mid-extraction still has no rollback). Files a newer payload no longer ships
; stay until uninstall, which removes them with the rest. A superseded package's *.dist-info
; stays beside its successor's until uninstall and reinstall (keeping %USERPROFILE%\Ouroboros).
; Setup refuses before touching anything when the target folder holds foreign files (such as
; an unpacked portable ZIP) or, on a first install, when the Start menu already has a foreign
; Ouroboros shortcut. It never closes a running Ouroboros: target binaries and unins*.dat are
; probed with an exclusive read/write OS open (fail-closed).
;
; Before touching anything, Setup and Uninstall acquire SetupChangingName exclusively
; (CreateMutexW + DllGetLastError <> 183) and release it once changes finish (ssPostInstall /
; usPostUninstall / Deinitialize*). A losing Setup refuses on the Preparing page; a losing
; Uninstall returns False from InitializeUninstall immediately with no dialog so Inno frees
; its exclusive unins*.dat handle at once. While the mutex exists, Ouroboros.exe exits 75
; immediately with no dialog and the shipped CLI prints a notice to stderr and exits 75
; (launcher_windows_runtime.refuse_while_setup_changes); portable copies are unaffected.
; Once nothing is in use, HoldStartPrograms holds Ouroboros.exe and the CLI's python.exe
; against new starts (Setup replaces them last after releasing each hold; Uninstall releases
; holds at usUninstall before deleting the shortcut and those two binaries first in reverse
; log order). It never touches %USERPROFILE%\Ouroboros, %LOCALAPPDATA%\Ouroboros\bin, or the
; sign-in startup choice.

#ifndef AppVersion
  #error Define AppVersion, AppNumericVersion and PayloadRoot: scripts\pack_windows_setup.ps1 does
#endif
#ifndef AppNumericVersion
  #error Define AppNumericVersion (x.y.z.0)
#endif
#ifndef PayloadRoot
  #error Define PayloadRoot, the collected Ouroboros folder
#endif

[Setup]
; Never change AppId: an installed copy is found and upgraded by it.
AppId={{6FB7704B-C2C5-4252-A5D0-BEC473FD007D}
AppName=Ouroboros
AppVersion={#AppVersion}
AppPublisher=Ouroboros
AppPublisherURL=https://ouroboros-agent.ai
AppSupportURL=https://github.com/razzant/ouroboros/issues
AppUpdatesURL=https://github.com/razzant/ouroboros/releases
VersionInfoVersion={#AppNumericVersion}
VersionInfoProductTextVersion={#AppVersion}
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\Ouroboros
DisableDirPage=yes
UsePreviousAppDir=no
AlwaysShowDirOnReadyPage=yes
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; The bundled Node.js and WebView2 desktop need Windows 10 or later.
MinVersion=10.0
; CloseApplications=yes would close processes in silent mode; keep it off and
; let PrepareToInstall / InitializeUninstall ask the owner to quit Ouroboros.
CloseApplications=no
RestartApplications=no
; Inno does not restore overwritten files on cancelled reinstall; disable normal
; Cancel once file copying starts while keeping Welcome/Ready/Preparing cancellable.
AllowCancelDuringInstall=no
SetupMutex=OuroborosSetup
UninstallDisplayName=Ouroboros
UninstallDisplayIcon={app}\Ouroboros.exe
SetupIconFile=..\..\assets\icon.ico
WizardStyle=modern dynamic
WizardImageFile=wizard.png
WizardImageFileDynamicDark=wizard-dark.png
WizardSmallImageFile=wizard-small.png
WizardSmallImageFileDynamicDark=wizard-small-dark.png
ShowLanguageDialog=no
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
SetupLogging=yes

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"

[CustomMessages]
AppRunning=Cannot access an Ouroboros file or folder right now (it may be in use or access was denied):%n%1%n%nIf Ouroboros is running, quit it with Quit Ouroboros in its notification-area menu, or resolve what blocks access, then click Retry. Setup never closes Ouroboros itself.
AppStillRunning=An Ouroboros file or folder is still in use or inaccessible, so nothing was changed. Close Ouroboros or resolve access to the install folder, then run Setup again.
UninstallAppRunning=Cannot access an Ouroboros file or folder right now (it may be in use or access was denied):%n%1%n%nIf Ouroboros is running, quit it with Quit Ouroboros in its notification-area menu, or resolve what blocks access, then click Retry. Uninstall never closes Ouroboros itself.
OnlyCanonicalDir=This installer installs Ouroboros for the current user only into%n%1%n%nTo keep Ouroboros in another folder, use the portable ZIP.
ForeignInstallDir=%1%n%nalready exists and was not installed by Ouroboros Setup (for example, it may hold an unpacked portable ZIP or your own files). Setup will not overwrite or delete it.%n%nMove or remove that folder and run Setup again, or keep using the portable ZIP.
ForeignShortcut=The Start menu already has an Ouroboros shortcut that Ouroboros Setup did not create:%n%1%n%nSetup will not replace or delete it. Remove or rename that shortcut (for example, one you made for the portable ZIP), then run Setup again.
InstallOrUninstallRunning=Another Ouroboros Setup or Uninstall is already changing this installation. Wait for it to finish, then try again.

[Files]
; Everything but the two programs the entries run comes first; each of those follows in its own entry, which releases
; its hold (HoldStartPrograms) just before replacing it, while the SetupChangingName mutex keeps both entries refusing.
Source: "{#PayloadRoot}\*"; Excludes: "\Ouroboros.exe,\_internal\python-standalone\python.exe"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#PayloadRoot}\_internal\python-standalone\python.exe"; DestDir: "{app}\_internal\python-standalone"; Flags: ignoreversion; BeforeInstall: ReleaseHeldProgram('_internal\python-standalone\python.exe')
Source: "{#PayloadRoot}\Ouroboros.exe"; DestDir: "{app}"; Flags: ignoreversion; BeforeInstall: ReleaseHeldProgram('Ouroboros.exe')

[Icons]
Name: "{userprograms}\Ouroboros"; Filename: "{app}\Ouroboros.exe"; AppUserModelID: "com.ouroboros.agent"

[Run]
Filename: "{app}\Ouroboros.exe"; Description: "{cm:LaunchProgram,Ouroboros}"; Flags: nowait postinstall skipifsilent

[Code]
const
  UninstallRegKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{6FB7704B-C2C5-4252-A5D0-BEC473FD007D}_is1';
  { Delphi System.SysUtils fmOpenReadWrite ($0002) or fmShareExclusive ($0010):
    opens an existing file with GENERIC_READ|GENERIC_WRITE and dwShareMode=0. }
  OpenExclusiveReadWrite = $0012;
  { fmOpenWrite ($0001) or fmShareDenyNone ($0040): GENERIC_WRITE sharing read and write,
    which the start hold below admits; Windows still refuses it for a running image. }
  OpenWriteShared = $0041;
  FileAttributeDirectory = $10;
  { The start hold: CreateFileW GENERIC_READ sharing only FILE_SHARE_WRITE ($2) and
    FILE_SHARE_DELETE ($4). Windows opens an image to run it with read/execute access,
    which this refuses; deleting needs delete sharing, which it grants. }
  GenericRead = $80000000;
  ShareWriteDelete = $00000006;
  OpenExisting = 3;
  InvalidHandle = -1;
  ErrorAlreadyExists = 183;
  { Held exclusively from before any filesystem check or mutation until nothing changes the installation any more:
    Setup and Uninstall exclude each other on it, and Ouroboros.exe / the shipped CLI consult it in their own
    processes before any work (launcher_windows_runtime._SETUP_CHANGING). }
  SetupChangingName = 'Local\Ouroboros {6FB7704B-C2C5-4252-A5D0-BEC473FD007D} changing';

var
  { One handle per StartProgram, 0 when none: taken by HoldStartPrograms, closed by ReleaseHold. }
  StartHolds: array[0..1] of NativeInt;
  { This run's exclusive handle to the SetupChangingName mutex, 0 when none. }
  SetupChanging: NativeInt;

function CreateFile(FileName: String; DesiredAccess, ShareMode: Cardinal; SecurityAttributes: NativeInt;
  CreationDisposition, FlagsAndAttributes: Cardinal; TemplateFile: NativeInt): NativeInt;
  external 'CreateFileW@kernel32.dll stdcall';
function CloseHandle(HeldObject: NativeInt): Integer;
  external 'CloseHandle@kernel32.dll stdcall';
function CreateNamedMutex(SecurityAttributes: NativeInt; InitialOwner: Integer; Name: String): NativeInt;
  external 'CreateMutexW@kernel32.dll stdcall';
procedure SetLastError(ErrCode: Cardinal);
  external 'SetLastError@kernel32.dll stdcall';

function CanonicalAppDir: String;
begin
  Result := ExpandConstant('{localappdata}\Programs\Ouroboros');
end;

{ True only when Dir can be enumerated and contains no entries besides . and .. }
function IsDirEmpty(const Dir: String): Boolean;
var
  FindRec: TFindRec;
begin
  Result := False;
  if FindFirst(AddBackslash(Dir) + '*', FindRec) then
  begin
    try
      Result := True;
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
        begin
          Result := False;
          Exit;
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;
end;

{ True only when HKCU uninstall metadata proves AppDir belongs to an earlier
  Ouroboros Setup install and both unins*.exe and its manifest .dat exist there.
  On True, UninstDat receives that manifest .dat path. }
function IsOwnedInstallDir(const AppDir: String; var UninstDat: String): Boolean;
var
  RecordedDir, RawUninstall, UninstExe: String;
begin
  Result := False;
  UninstDat := '';
  if not RegQueryStringValue(HKEY_CURRENT_USER, UninstallRegKey, 'Inno Setup: App Path', RecordedDir) then
  begin
    if not RegQueryStringValue(HKEY_CURRENT_USER, UninstallRegKey, 'InstallLocation', RecordedDir) then
      Exit;
  end;
  if CompareText(RemoveBackslashUnlessRoot(Trim(RecordedDir)), AppDir) <> 0 then
    Exit;
  if not RegQueryStringValue(HKEY_CURRENT_USER, UninstallRegKey, 'UninstallString', RawUninstall) then
    Exit;
  UninstExe := RemoveQuotes(Trim(RawUninstall));
  if UninstExe = '' then
    Exit;
  if CompareText(RemoveBackslashUnlessRoot(ExtractFileDir(UninstExe)), AppDir) <> 0 then
    Exit;
  if not WildcardMatch(Lowercase(ExtractFileName(UninstExe)), 'unins*.exe') then
    Exit;
  Result := FileExists(UninstExe) and FileExists(ChangeFileExt(UninstExe, '.dat'));
  if Result then
    UninstDat := ChangeFileExt(UninstExe, '.dat');
end;

{ True when FilePath exists and cannot be opened exclusively for read/write
  (active image/DLL mapping, open handle, or inaccessible state). Fail-closed:
  any exception on an existing file counts as busy, never as free. }
function IsFileLocked(const FilePath: String): Boolean;
var
  Stream: TFileStream;
begin
  Result := False;
  if not FileExists(FilePath) then
    Exit;
  try
    Stream := TFileStream.Create(FilePath, OpenExclusiveReadWrite);
    try
    finally
      Stream.Free;
    end;
  except
    Log('Target file is busy or inaccessible (' + FilePath + '): ' + GetExceptionMessage);
    Result := True;
  end;
end;

{ Returns the first locked python3*.dll inside Dir (or Dir itself if Dir exists
  but cannot be enumerated — fail-closed); returns '' when none is locked. }
function BusyPythonDllInDir(const Dir: String): String;
var
  FindRec: TFindRec;
  Path: String;
begin
  Result := '';
  if not DirExists(Dir) then
    Exit;
  if not FindFirst(AddBackslash(Dir) + '*', FindRec) then
  begin
    Log('Could not enumerate ' + Dir + ' while checking locked runtime DLLs.');
    Result := Dir;
    Exit;
  end;
  try
    repeat
      if ((FindRec.Attributes and FileAttributeDirectory) = 0) and
         WildcardMatch(Lowercase(FindRec.Name), 'python3*.dll') then
      begin
        Path := AddBackslash(Dir) + FindRec.Name;
        if IsFileLocked(Path) then
        begin
          Result := Path;
          Exit;
        end;
      end;
    until not FindNext(FindRec);
  finally
    FindClose(FindRec);
  end;
end;

{ Returns the first target runtime binary, Python DLL, or optional ExtraFile (Setup
  passes unins*.dat; Uninstall passes '' so it never trips on itself) in Dir that holds
  an OS lock, or '' when every existing target can be opened exclusively. }
function BusyFileInDir(const Dir, ExtraFile: String): String;
var
  Candidates: array[0..7] of String;
  Base, Path: String;
  I: Integer;
begin
  Result := '';
  if (ExtraFile <> '') and IsFileLocked(ExtraFile) then
  begin
    Result := ExtraFile;
    Exit;
  end;
  Base := AddBackslash(Dir);
  Candidates[0] := Base + 'Ouroboros.exe';
  Candidates[1] := Base + '_internal\python-standalone\python.exe';
  Candidates[2] := Base + '_internal\python-standalone\pythonw.exe';
  Candidates[3] := Base + '_internal\node-standalone\node.exe';
  Candidates[4] := Base + '_internal\ripgrep-standalone\rg.exe';
  Candidates[5] := Base + '_internal\betterleaks-standalone\bin\betterleaks.exe';
  Candidates[6] := Base + '_internal\git-standalone\cmd\git.exe';
  Candidates[7] := Base + '_internal\git-standalone\mingw64\bin\git.exe';
  for I := 0 to GetArrayLength(Candidates) - 1 do
  begin
    Path := Candidates[I];
    if IsFileLocked(Path) then
    begin
      Result := Path;
      Exit;
    end;
  end;
  Result := BusyPythonDllInDir(Base + '_internal\python-standalone');
  if Result <> '' then
    Exit;
  Result := BusyPythonDllInDir(Base + '_internal');
end;

{ True once every target file in Dir is free; False when the owner cancels or
  Setup/Uninstall is silent (SuppressibleMsgBox defaults to IDCANCEL). }
function WaitUntilQuit(const Dir, ExtraFile, Message: String): Boolean;
var
  Busy: String;
begin
  Result := True;
  Busy := BusyFileInDir(Dir, ExtraFile);
  while Busy <> '' do
  begin
    if SuppressibleMsgBox(FmtMessage(Message, [Busy]), mbError, MB_RETRYCANCEL, IDCANCEL) <> IDRETRY then
    begin
      Result := False;
      Exit;
    end;
    Busy := BusyFileInDir(Dir, ExtraFile);
  end;
end;

{ True while Windows maps FilePath as a running image: a write open the start hold admits is
  still refused for an image in use. Fail-closed: any exception counts as running. }
function IsRunningImage(const FilePath: String): Boolean;
var
  Stream: TFileStream;
begin
  Result := False;
  try
    Stream := TFileStream.Create(FilePath, OpenWriteShared);
    try
    finally
      Stream.Free;
    end;
  except
    Log('Target image is running or inaccessible (' + FilePath + '): ' + GetExceptionMessage);
    Result := True;
  end;
end;

{ The programs a start runs: Ouroboros.exe from the Start menu or sign-in startup, python.exe from the CLI. }
function StartProgram(const I: Integer): String;
begin
  if I = 0 then
    Result := 'Ouroboros.exe'
  else
    Result := '_internal\python-standalone\python.exe';
end;

procedure ReleaseHold(const I: Integer);
begin
  if StartHolds[I] <> 0 then
  begin
    CloseHandle(StartHolds[I]);
    StartHolds[I] := 0;
    Log('Start hold released: ' + StartProgram(I));
  end;
end;

procedure ReleaseHolds;
var
  I: Integer;
begin
  for I := 0 to GetArrayLength(StartHolds) - 1 do
    ReleaseHold(I);
end;

{ BeforeInstall of a held program's own [Files] entry: every other file is in place and Setup replaces this one next. }
procedure ReleaseHeldProgram(RelPath: String);
var
  I: Integer;
begin
  for I := 0 to GetArrayLength(StartHolds) - 1 do
    if CompareText(StartProgram(I), RelPath) = 0 then
      ReleaseHold(I);
end;

{ Before any change: True once nothing Setup or Uninstall would change is in use and every existing StartProgram is
  held so that Windows refuses to run it; False when the owner cancels, and at once when silent. A start between the
  busy probe and the holds shows up as a refused hold or a running image and is asked about again. }
function HoldStartPrograms(const AppDir, ExtraFile, Message: String): Boolean;
var
  I: Integer;
  Path, Refused: String;
begin
  repeat
    Result := WaitUntilQuit(AppDir, ExtraFile, Message);
    if not Result then
      Exit;
    Refused := '';
    for I := 0 to GetArrayLength(StartHolds) - 1 do
    begin
      Path := AddBackslash(AppDir) + StartProgram(I);
      if (Refused = '') and FileExists(Path) then
      begin
        StartHolds[I] := CreateFile(Path, GenericRead, ShareWriteDelete, 0, OpenExisting, 0, 0);
        if StartHolds[I] = InvalidHandle then
          StartHolds[I] := 0;
        if (StartHolds[I] = 0) or IsRunningImage(Path) then
          Refused := Path
        else
          Log('Start hold taken: ' + StartProgram(I));
      end;
    end;
    if Refused = '' then
      Exit;
    ReleaseHolds;
    Result := SuppressibleMsgBox(FmtMessage(Message, [Refused]), mbError, MB_RETRYCANCEL, IDCANCEL) = IDRETRY;
  until not Result;
end;

{ Atomically acquires SetupChangingName exclusively (DllGetLastError <> ERROR_ALREADY_EXISTS).
  On collision, closes the handle and returns False immediately with no dialog so a losing
  Uninstall releases Inno's exclusive unins*.dat handle at once. }
function TryAcquireChanging: Boolean;
var
  Handle: NativeInt;
  Err: LongInt;
begin
  Result := SetupChanging <> 0;
  if Result then
    Exit;
  SetLastError(0);
  Handle := CreateNamedMutex(0, 0, SetupChangingName);
  Err := DllGetLastError;
  if (Handle = 0) or (Err = ErrorAlreadyExists) then
  begin
    if Handle <> 0 then
      CloseHandle(Handle);
    Log('Refusing concurrent change (mutex Win32 error ' + IntToStr(Err) + ').');
    Exit;
  end;
  SetupChanging := Handle;
  Result := True;
end;

procedure ClearChanging;
begin
  if SetupChanging <> 0 then
    CloseHandle(SetupChanging);
  SetupChanging := 0;
end;

procedure ReleaseChangeState;
begin
  ReleaseHolds;
  ClearChanging;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  AppDir, Shortcut, UninstDat: String;
begin
  Result := '';
  AppDir := RemoveBackslashUnlessRoot(ExpandConstant('{app}'));
  if CompareText(AppDir, RemoveBackslashUnlessRoot(CanonicalAppDir)) <> 0 then
  begin
    Result := FmtMessage(CustomMessage('OnlyCanonicalDir'), [CanonicalAppDir]);
    Exit;
  end;
  if not TryAcquireChanging then
  begin
    Result := CustomMessage('InstallOrUninstallRunning');
    Exit;
  end;
  UninstDat := '';
  if FileExists(AppDir) and not DirExists(AppDir) then
    Result := FmtMessage(CustomMessage('ForeignInstallDir'), [AppDir])
  else if not DirExists(AppDir) or IsDirEmpty(AppDir) then
  begin
    Shortcut := ExpandConstant('{userprograms}\Ouroboros.lnk');
    if FileExists(Shortcut) then
      Result := FmtMessage(CustomMessage('ForeignShortcut'), [Shortcut]);
  end
  else if not IsOwnedInstallDir(AppDir, UninstDat) then
    Result := FmtMessage(CustomMessage('ForeignInstallDir'), [AppDir])
  else if not HoldStartPrograms(AppDir, UninstDat, CustomMessage('AppRunning')) then
    Result := CustomMessage('AppStillRunning');
  if Result <> '' then
    ReleaseChangeState;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    ReleaseChangeState;
end;

procedure DeinitializeSetup;
begin
  ReleaseChangeState;
end;

function InitializeUninstall: Boolean;
begin
  Result := False;
  if not TryAcquireChanging then
    Exit;
  Result := HoldStartPrograms(RemoveBackslashUnlessRoot(ExpandConstant('{app}')), '', CustomMessage('UninstallAppRunning'));
  if not Result then
    ReleaseChangeState;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
    ReleaseHolds
  else if CurUninstallStep = usPostUninstall then
    ClearChanging;
end;

procedure DeinitializeUninstall;
begin
  ReleaseChangeState;
end;
