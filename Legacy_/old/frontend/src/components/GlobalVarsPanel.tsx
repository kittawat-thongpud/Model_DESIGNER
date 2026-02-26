/**
 * GlobalVarsPanel — sidebar panel for creating & managing global config variables.
 * Supports: bool, float, int, str, selector types.
 * Variables can be referenced by block params via $var_name syntax.
 */
import { useState } from 'react';
import { useDesignerStore } from '../store/designerStore';
import type { GlobalVariable, GlobalVarType } from '../types';

const TYPE_ICONS: Record<GlobalVarType, string> = {
  bool: '🔘',
  float: '🔢',
  int: '#️⃣',
  str: '📝',
  selector: '📋',
};

const TYPE_COLORS: Record<GlobalVarType, string> = {
  bool: '#22c55e',
  float: '#3b82f6',
  int: '#a855f7',
  str: '#f59e0b',
  selector: '#ec4899',
};

const DEFAULT_VALUES: Record<GlobalVarType, boolean | number | string> = {
  bool: true,
  float: 0.5,
  int: 1,
  str: '',
  selector: '',
};

let varCounter = 0;

export default function GlobalVarsPanel() {
  const globalVars = useDesignerStore((s) => s.globalVars);
  const addGlobalVar = useDesignerStore((s) => s.addGlobalVar);
  const updateGlobalVar = useDesignerStore((s) => s.updateGlobalVar);
  const deleteGlobalVar = useDesignerStore((s) => s.deleteGlobalVar);

  const [showAdd, setShowAdd] = useState(false);
  const [newName, setNewName] = useState('');
  const [newType, setNewType] = useState<GlobalVarType>('float');
  const [editingId, setEditingId] = useState<string | null>(null);

  const handleAdd = () => {
    const name = newName.trim().replace(/\s+/g, '_').toLowerCase();
    if (!name) return;
    if (globalVars.some((v) => v.name === name)) return; // duplicate

    const gvar: GlobalVariable = {
      id: `gvar_${++varCounter}_${Date.now()}`,
      name,
      type: newType,
      value: DEFAULT_VALUES[newType],
      options: newType === 'selector' ? ['option_a', 'option_b'] : undefined,
      description: '',
    };
    addGlobalVar(gvar);
    setNewName('');
    setShowAdd(false);
  };

  const handleValueChange = (gvar: GlobalVariable, rawValue: string) => {
    let parsed: boolean | number | string;
    switch (gvar.type) {
      case 'bool':
        parsed = rawValue === 'true';
        break;
      case 'float':
        parsed = parseFloat(rawValue) || 0;
        break;
      case 'int':
        parsed = parseInt(rawValue) || 0;
        break;
      default:
        parsed = rawValue;
    }
    updateGlobalVar(gvar.id, { value: parsed });
  };

  const renderValueEditor = (gvar: GlobalVariable) => {
    if (gvar.type === 'bool') {
      return (
        <button
          className={`gvp-toggle ${gvar.value ? 'on' : 'off'}`}
          onClick={() => updateGlobalVar(gvar.id, { value: !gvar.value })}
        >
          {gvar.value ? 'true' : 'false'}
        </button>
      );
    }

    if (gvar.type === 'selector') {
      return (
        <select
          className="gvp-select"
          value={String(gvar.value)}
          onChange={(e) => updateGlobalVar(gvar.id, { value: e.target.value })}
        >
          {(gvar.options ?? []).map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      );
    }

    return (
      <input
        className="gvp-input"
        type={gvar.type === 'str' ? 'text' : 'number'}
        step={gvar.type === 'float' ? '0.01' : '1'}
        value={String(gvar.value)}
        onChange={(e) => handleValueChange(gvar, e.target.value)}
      />
    );
  };

  return (
    <div className="gvp-container">
      <div className="gvp-header">
        <span className="gvp-title">🌐 Global Variables</span>
        <span className="gvp-count">{globalVars.length}</span>
        <button className="gvp-add-btn" onClick={() => setShowAdd(!showAdd)} title="Add variable">
          {showAdd ? '✕' : '+'}
        </button>
      </div>

      {/* Add new variable form */}
      {showAdd && (
        <div className="gvp-add-form">
          <input
            className="gvp-name-input"
            type="text"
            placeholder="variable_name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
            autoFocus
          />
          <div className="gvp-type-picker">
            {(Object.keys(TYPE_ICONS) as GlobalVarType[]).map((t) => (
              <button
                key={t}
                className={`gvp-type-btn ${newType === t ? 'active' : ''}`}
                style={{ '--type-color': TYPE_COLORS[t] } as React.CSSProperties}
                onClick={() => setNewType(t)}
                title={t}
              >
                {TYPE_ICONS[t]} {t}
              </button>
            ))}
          </div>
          <button className="gvp-create-btn" onClick={handleAdd} disabled={!newName.trim()}>
            Create Variable
          </button>
        </div>
      )}

      {/* Variable list */}
      <div className="gvp-list">
        {globalVars.length === 0 && !showAdd && (
          <div className="gvp-empty">
            No global variables yet.<br />
            <span className="gvp-empty-hint">Click + to create one.</span>
          </div>
        )}

        {globalVars.map((gvar) => (
          <div
            key={gvar.id}
            className={`gvp-item ${editingId === gvar.id ? 'editing' : ''}`}
          >
            <div className="gvp-item-header">
              <span className="gvp-item-type" style={{ color: TYPE_COLORS[gvar.type] }}>
                {TYPE_ICONS[gvar.type]}
              </span>
              <span className="gvp-item-name">${gvar.name}</span>
              <span className="gvp-item-type-badge" style={{ borderColor: TYPE_COLORS[gvar.type], color: TYPE_COLORS[gvar.type] }}>
                {gvar.type}
              </span>
              <button
                className="gvp-edit-btn"
                onClick={() => setEditingId(editingId === gvar.id ? null : gvar.id)}
                title="Edit"
              >
                ✎
              </button>
              <button
                className="gvp-delete-btn"
                onClick={() => deleteGlobalVar(gvar.id)}
                title="Delete variable"
              >
                🗑
              </button>
            </div>

            <div className="gvp-item-value-row">
              {renderValueEditor(gvar)}
            </div>

            {/* Expanded editor for selector options */}
            {editingId === gvar.id && (
              <div className="gvp-item-edit">
                <div className="gvp-edit-field">
                  <label>Description</label>
                  <input
                    type="text"
                    value={gvar.description ?? ''}
                    onChange={(e) => updateGlobalVar(gvar.id, { description: e.target.value })}
                    placeholder="What this variable controls..."
                  />
                </div>
                {gvar.type === 'selector' && (
                  <div className="gvp-edit-field">
                    <label>Options (comma-separated)</label>
                    <input
                      type="text"
                      value={(gvar.options ?? []).join(', ')}
                      onChange={(e) => {
                        const options = e.target.value.split(',').map((s) => s.trim()).filter(Boolean);
                        updateGlobalVar(gvar.id, { options });
                      }}
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
