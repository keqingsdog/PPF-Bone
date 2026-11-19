# f740 → c0d stable plugin merge (2026-10-26)

This branch merged the stable fixes by function against the supplied f740 base and the existing local plugin. The old full addon did not replace the current tree. All 21 stable product paths are included; the experimental external-PC2 Undo branch is excluded. No crates, native runtime, backend build route or engine math changed.

## Proven current defects and fixes

- Object identity/groups: hot duplicate+rename selected the wrong object; cold and linked ambiguity could resolve to a different owner; later batch/API Clear conflicts changed valid prefixes. The old automatic Wireframe also removed edge-only ROD geometry. Real Windows before/after: 7 failing conditions plus one valid linked control → 8 passing conditions. A separate genuine file save and fresh-process reopen passed unique and ambiguous UUID cases, including actual load-hook reset.
- Profiles/capture: a NUL configuration partially changed state; changing the effective frame count after sampling still committed old buffers. Configuration is validated before mutation, selected profiles keep name identity, capture range/member/origin proofs are checked before publish/export, and interrupted paired commits preserve the previous pair. Real Pin/STATIC capture, CBOR, invalidation/recapture, save/reopen and injected second-replace failure passed.
- Bake/Clear/Curve/numeric: old Curve Save As and known ALIGNED author intent were not durably retained; invalid numeric data could publish part of a multi-target PC2 frame; foreign Scene ownership changes were not stopped early enough. Whole-frame numeric admission, ownership preflight, bound Curve output and known handle-intent retention were migrated. Finite negative mapping coefficients remain valid. Existing normal Mesh/Curve author-Action Bake behavior was already equivalent and is preserved. Same 32-check suite: 13 failing before conditions → 32/32 after. Four Curve lifecycle cases, existing f740 eight failure cases and nine independent reopen records passed.

## Preserved local behavior

New RIGID/LOW, SOFT/HIGH/500 MPa and density1 defaults; true-type compact bone UI; multi-selection changes; first-delete geometry-source/PRESERVE lineage; T cycling and E mesh editing; GPU snapshot lifecycle; static disabled-middle bridge/current-frame FK; public product-profile selector and all existing backend paths remain. Final installed-package tests explicitly retained the default, bridge and frozen-scene deletion/recreation behavior.

## Final installed Windows evidence

The candidate was actually installed/enabled by Blender Extension Manager in fresh private profiles, using bl_ext.user_default.ppf_contact_solver and its bundled Windows cp313 cbor2 wheel, without source/dependency-path fallback. All six focused suites passed naturally with exit0: identity, genuine cold reopen, capture/CBOR, Bake/numeric/Clear, Curve lifecycle, and local bone-feature retention. FINAL_RESULT.json in the merge evidence folder points to the actual per-suite results.

Two first installation observers were corrected without product changes: a Curve fixture required an optional source-root environment variable; the old deletion probe read a user file that had since changed. The latter now uses the already-frozen private Scene, whose three proxies genuinely have DEFAULTS provenance. Original failed logs remain; they are not reported as product fixes.

## Boundaries and usage

Evidence is Windows Blender5.1.2 programmatic/background RNA/operator/PC2/Action/save-reopen validation. No natural GUI click/visual acceptance, solver-accuracy acceptance, new native simulation or all-platform rerun is claimed. The historical Linux649 suite was not rerun. Existing f740 fault tests use explicit fault injection, not new real OS permission failures.

Unknown historical Curve handle types cannot be reconstructed; external PC2 native Undo remains outside stable scope. Legacy captured input lacking a valid proof must be explicitly recaptured. Paired-file/numeric staging is not general filesystem/RNA atomicity. Static disabled-middle restrictions and the original unconfirmed Undo AV boundary remain unchanged.

The installed addon files and existing installer/source share are updated in place. Save the current project and restart Blender to load the Python changes; then recapture stale inputs if requested and transfer normally. The native solver does not need recompilation for this merge.

## Reproduction pointers

Local evidence root: <historical-evidence-root>/plugin-cloud-c0d-merge-20261026. Group reports are identity-groups/RESULT.md, capture-profiles/REPORT.md and bake-curve/REPORT.md; final results are FINAL_RESULT.json. The group scripts accept an addon module name and disposable output paths. They are test-only and are not runtime/build dependencies or personal assets. The supplied stable source retains the upstream tests; no archived private environments are needed to use or build the product.

## Supplement: actual hidden GUI event-loop cleanup (2026-10-26)

A further private Windows Blender5.1.2 process installed the same candidate, entered the real non-background GUI main loop while hidden, and reopened the generated cold-identity fixture. The real registered tick invoked the original cleanup twice: once for a collision (one warning, preserved author/groups, stopped timer and no idle retries), then once after a real depsgraph update in a healthy Scene (removed only a stale member, retained valid members and cleared the warning). Eleven bounded checks passed, stdout ended normally and stderr was empty. This is actual timer/main-loop evidence, not natural clicking or visual UI acceptance. No product or original project was changed.

Two earlier observation attempts remain recorded: startup context was unavailable in the first; the second had not generated the required depsgraph event for the positive control. The first also hit a native AV after immediately quitting the incomplete startup context; its cause is unconfirmed and is not assigned to the product, erased, or claimed fixed by the later pass. The original user Undo AV likewise remains unconfirmed.

Evidence: plugin-cloud-c0d-merge-20261026/deferred-cleanup/RESULT.md and attempt3/receipt.json; reviewer STATUS PASS for this bounded path. No additional product patch, installer rebuild, solver run or broad matrix was performed.

## Startup/immediate-quit AV isolation

The same minimal private startup open_mainfile then immediate-quit flow reproduced wm_exit_schedule_delayed -> wm_exit_blender_exec read0xF0 both without PPF and with the installed candidate. PPF is not necessary for this specific AV; this does not explain the original Undo AV. Both crash-handling processes stayed live beyond45s and their own Popen watchdogs terminated them with exit1. Those are termination codes, not natural success or native AV exit codes. No product changed. Evidence: plugin-cloud-c0d-merge-20261026/startup-av-isolation/RESULT.md and pair-result.json.

## Current-version mode/Undo self-test supplement

No product was changed. The delivered candidate was tested in private hidden Blender5.1.2 GUI processes using actual native transforms, automatic Undo steps, natural FramePump updates and real GPU drawing. Object-scale cancellation and Pose Undo/Redo restored the independent placement/geometry oracle. A first process had no GPU draw and was not accepted as draw coverage. A second Object-baseline Undo restored 3 m to2 m correctly but was not mislabelled as Edit-BMesh Undo.

A final narrow Edit-baseline run (PID42560) used one setup push in Edit mode, native Resize2→3 m, a later natural derived mesh swap, and one genuine Undo back to EDIT_ARMATURE. EditBone was2 m while committed data.bones and mesh were still3 m. Exiting Edit then allowed the original FramePump to restore all three proxies to2 m, axis endpoints[0,2] and durable reference2 m. Position/orientation/relative-scale/endpoint errors were0; maximum local mesh error5.96e-8 was below the2e-5 threshold. Actual draw199, GPU batch1146, geometry calls from Draw or during identity rebuilding0, natural process exit0. No post-swap push or manual product repair created the result.

Evidence: plugin-cloud-c0d-merge-20261026/mode-disorder/RESULT.md and exact-edit-undo/result.json/process.json. The bounded path passed review after correcting an initially overstated all-zero mesh-error sentence. This does not prove the original user's unrecorded AV is eliminated. The previously delivered plugin remains unchanged; Linux can continue checking that same version. No additional private test loop is planned without a new concrete failure.

## Cause resolved and test-harness workaround verified

This supersedes the earlier unconfirmed cause for the startup/immediate-quit AV only. Official Blender5.1.2 wm_exit_schedule_delayed obtains CTX_wm_window(C), then accesses win->runtime->modalhandlers without testing win for null. The official Context API documents that open_mainfile can invalidate the context Window and that callers should reacquire a Window and use temp_override. Local receipts confirm window=None immediately after file loading in both PPF-disabled and enabled controls; both previously crashed in this exact native function. Source: https://github.com/blender/blender/blob/v5.1.2/source/blender/windowmanager/intern/wm_init_exit.cc#L436-L448. API: https://docs.blender.org/api/5.0/bpy.types.Context.html.

The trigger was our automation script immediately invoking the quit operator from the cleared startup context. The script was corrected to yield to a natural timer, reacquire its own current Window and quit under a valid Window override. Same candidate installed/enabled, PID51376: naturally exited0, no timeout, stderr empty/no AV, quit returned FINISHED. Evidence: safe-quit/process.json/context.json and SAFE_QUIT_RESULT.md. This changes only our test harness; the delivered PPF plugin, solver and Blender executable were not modified or rebuilt.

A Blender-source hardening fix would reject or appropriately handle a null Window before that dereference; no upstream-fix release was verified or claimed. The user's historical manual Undo AV remains a separate unconfirmed failure and is not explained away by this startup finding.
