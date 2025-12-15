# FLAMINGO simulations merger trees module

## Reading merger trees

TODO!

## Constructing merger trees

### Input halo catalogue

HBT-HERONS already outputs merger tree information, but organized in a
way that is not convenient for searching for progenitors and
descendants. This module can generate a merger tree file which
contains halo information stored in depth first order for faster
progenitor and descendant queries.

The starting point is a SOAP halo catalogue for each snapshot,
including the TrackId, SinkTrackId and DescendantTrackId fields from
HBT-HERONS.

### Method

  * Read in SOAP halos for all snapshots
  * Determine the "end" of each HBT track:
    * The first or last snapshot at which the halo becomes unresolved?
    * Or the first snapshot where the subhalo has sunk?
    * Might want to provide two versions of the trees
  * Label (or discard?) tracks which have ended
  * Determine descendant index for each halo
    * Same TrackId if track hasn't ended
    * SInkTrackId or DescendantTrackId if track has ended
  * Determine final descendant for each halo
  * Move all halos with the same final descendant to the same MPI rank
  * Compute depth first index within each tree
  * Make these indexes unique between trees
  * Assign end main branch and last progenitor IDs

Also need to pass through SOAP index for each halo for cross referencing.
Pass through a few other properties too?

Write out one big file per simulation?
