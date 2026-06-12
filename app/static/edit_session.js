/**
 * Session Edit: "Add Set" button handler
 * 
 * When a user clicks "Add set" on an exercise block, append a new row to the
 * `.sets-grid` with the next set number, update the `extra_sets-EX-ID` hidden
 * input, and re-index all subsequent field names.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Find all "Add set" buttons
    document.querySelectorAll('.add-set-btn').forEach(btn => {
        btn.addEventListener('click', e => {
            e.preventDefault();
            addSetToExercise(btn);
        });
    });
});

/**
 * Add a new set row to an exercise block.
 * @param {HTMLElement} btn - The "Add set" button element
 */
function addSetToExercise(btn) {
    const exerciseBlock = btn.closest('.exercise-block');
    const exId = btn.dataset.exId;
    const weightPlaceholder = btn.dataset.weightPlaceholder;

    const grid = exerciseBlock.querySelector('.sets-grid');
    const isBw = exerciseBlock.querySelector(`input[name^="weight-${exId}-"]`).placeholder === 'Added kg';

    // Find the highest set number currently in the grid
    const existingSetInputs = Array.from(grid.querySelectorAll(`input[name^="reps-${exId}-"]`));
    let maxSetNum = 0;
    existingSetInputs.forEach(input => {
        const match = input.name.match(new RegExp(`reps-${exId}-(\\d+)`));
        if (match) {
            const setNum = parseInt(match[1], 10);
            if (setNum > maxSetNum) maxSetNum = setNum;
        }
    });

    const newSetNum = maxSetNum + 1;

    // Create the new row HTML
    const newSetNum_str = newSetNum.toString();
    const colCount = grid.style.gridTemplateColumns.split(' ').length || 3;
    const rowHTML = `
    <div class="set-num">${newSetNum_str}</div>
    <input type="number" name="reps-${exId}-${newSetNum_str}" min="0" placeholder="0" />
    <input type="number" step="0.5" name="weight-${exId}-${newSetNum_str}" min="0" placeholder="${weightPlaceholder}" />
    <label class="remove-set" title="Remove this set">
      <input type="checkbox" name="remove-${exId}-${newSetNum_str}" />
      <span aria-hidden="true">✕</span>
    </label>
  `;

    // Append to the grid
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = rowHTML;
    Array.from(tempDiv.children).forEach(child => grid.appendChild(child));

    // Increment the extra_sets hidden input
    const extraSetsInput = exerciseBlock.querySelector('.extra-sets-input');
    if (extraSetsInput) {
        extraSetsInput.value = (parseInt(extraSetsInput.value, 10) || 0) + 1;
    }
}
