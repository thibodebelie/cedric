param(
    [switch]$DryRun,
    [switch]$ContinueOnError
)

$python = 'python'
$script = Join-Path $PSScriptRoot 'run_all.py'

$args = @()
if ($DryRun) { $args += '--dry-run' }
if ($ContinueOnError) { $args += '--continue-on-error' }

& $python $script @args
