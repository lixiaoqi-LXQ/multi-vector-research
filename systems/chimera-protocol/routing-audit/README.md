# Chimera routing-dataset scientific isolation

Seven complete legacy indexes from the fixed affected inventory now contain `ROUTING_DATASET_UNVERIFIED.json`. One listed path is an assignment cache without six index payloads and was skipped. The markers retain old evidence while preventing future scientific acceptance. Six payload file stats and every preexisting JSON receipt SHA were checked unchanged. See `quarantine-receipt.json`.

Future `run_search_grid.py` and `accept_run.py` reject a marked index before accessing query inputs or running GPU code. Every new search requires these explicit spec fields:

```
"routing_dataset_audit_receipt": "/absolute/accepted/result.json",
"routing_dataset_audit_sha256": "<exact receipt SHA256>"
```

The guard requires a passing post-encode/save semantic audit: correct cluster count/D, seed42 FhtKac rotation, equal expected/actual dataset digests, zero FP32 bit mismatches, loaded device dataset, and the exact current centroids.carga SHA. It freezes this validation and guard-source SHA into the run receipt, rechecks each round and verifies the same binding before acceptance. A derived8 index can inherit fixed4's audit only with the same routing graph inode and an unquarantined source4; this check does not rehash multi-GB graphs each round.

Fresh4/fresh8/branchA/branchB and the fixed-source derived8 have approved sidecars `ROUTING_DATASET_AUDIT.json`. `approved-owned-device-spec-fields.json` provides exact paths and approved fields; derived8 also includes its explicit tool/qualification approval. `audit-sidecar-publication.json` records unchanged prior receipts and payload stats. The new derived8 build metadata was published through the previously reviewed adapter; its frozen requantization receipt and payloads remain intact.

The new `owned-fixed-source-quantizer-qualification.json` distinguishes current owned-source positive all-byte/inode checks from inherited negative guard tests of the exact unchanged tool/wrapper. It claims no inherited routing validity from old indexes. The routing audit supplies the independent graph acceptance.

CPU validation:14 cases passed, covering successful audits, quarantine rejection, missing/mutated receipts, wrong graph/shape/seed/memory type, semantic mismatch, same-graph derived inheritance and both CLI entry points rejecting before execution. Five real owned-device audit/derived-identity checks passed. `routing-guard-freeze.json` freezes source and evidence hashes and the actual20-thread OpenMP budget. Original runner functions except the intended main/identity_checks hooks are AST-identical; active checkpoint sources/builders and old run source identities were not modified.

For fullLoTTE/MS, the generic auditor at `chimera-checkpoint/full-dataset-audit` must inspect that full index's saved routing dataset against its own cached sampled-centroid matrix. Prefix8192 receipts are rejected for2^21/2^22 clusters. Fill the full search spec only after its full semantic receipt passes; source/bitwidth/GT identity and derived approval remain independently required.
