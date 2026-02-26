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
          className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
            gvar.value 
              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' 
              : 'bg-slate-800 text-slate-400 border border-slate-700'
          }`}
          onClick={() => updateGlobalVar(gvar.id, { value: !gvar.value })}
        >
          {gvar.value ? 'true' : 'false'}
        </button>
      );
    }

    if (gvar.type === 'selector') {
      return (
        <select
          className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-sm text-white focus:outline-none focus:border-indigo-500"
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
        className="w-full bg-slate-950 border border-slate-800 rounded px-2 py-1 text-sm text-white focus:outline-none focus:border-indigo-500"
        type={gvar.type === 'str' ? 'text' : 'number'}
        step={gvar.type === 'float' ? '0.01' : '1'}
        value={String(gvar.value)}
        onChange={(e) => handleValueChange(gvar, e.target.value)}
      />
    );
  };

  return (
    <div className="flex flex-col h-full bg-slate-900/50">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <span className="text-lg">🌐</span>
          <span className="text-sm font-semibold text-white">Global Variables</span>
          <span className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-400">{globalVars.length}</span>
        </div>
        <button 
          className="w-6 h-6 flex items-center justify-center rounded hover:bg-slate-700 text-slate-400 hover:text-white transition-colors text-lg"
          onClick={() => setShowAdd(!showAdd)} 
          title="Add variable"
        >
          {showAdd ? '✕' : '+'}
        </button>
      </div>

      {/* Add new variable form */}
      {showAdd && (
        <div className="p-3 border-b border-slate-800 space-y-3">
          <input
            className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-indigo-500"
            type="text"
            placeholder="variable_name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
            autoFocus
          />
          <div className="flex gap-2">
            {(Object.keys(TYPE_ICONS) as GlobalVarType[]).map((t) => (
              <button
                key={t}
                className={`px-2 py-1 rounded text-xs border transition-colors ${
                  newType === t 
                    ? 'border-indigo-500 bg-indigo-500/10 text-indigo-400' 
                    : 'border-slate-700 hover:border-slate-600 text-slate-400'
                }`}
                style={{ borderColor: newType === t ? TYPE_COLORS[t] : undefined, color: newType === t ? TYPE_COLORS[t] : undefined }}
                onClick={() => setNewType(t)}
                title={t}
              >
                <span className="mr-1">{TYPE_ICONS[t]}</span>
                {t}
              </button>
            ))}
          </div>
          <button 
            className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleAdd} 
            disabled={!newName.trim()}
          >
            Create Variable
          </button>
        </div>
      )}

      {/* Variable list */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {globalVars.length === 0 && !showAdd && (
          <div className="text-center text-slate-500 text-sm py-8">
            No global variables yet.<br />
            <span className="text-xs text-slate-600">Click + to create one.</span>
          </div>
        )}

        {globalVars.map((gvar) => (
          <div
            key={gvar.id}
            className={`bg-slate-950 border rounded-lg p-3 space-y-2 ${
              editingId === gvar.id ? 'border-indigo-500/30' : 'border-slate-800'
            }`}
          >
            <div className="flex items-center gap-2">
              <span style={{ color: TYPE_COLORS[gvar.type] }}>{TYPE_ICONS[gvar.type]}</span>
              <span className="font-mono text-sm text-white font-medium">${gvar.name}</span>
              <span 
                className="px-1.5 py-0.5 rounded text-[10px] border ml-auto"
                style={{ borderColor: TYPE_COLORS[gvar.type], color: TYPE_COLORS[gvar.type] }}
              >
                {gvar.type}
              </span>
              <button
                className="text-slate-500 hover:text-white text-sm"
                onClick={() => setEditingId(editingId === gvar.id ? null : gvar.id)}
                title="Edit"
              >
                ✎
              </button>
              <button
                className="text-slate-500 hover:text-red-400 text-sm"
                onClick={() => deleteGlobalVar(gvar.id)}
                title="Delete variable"
              >
                🗑
              </button>
            </div>

            <div>{renderValueEditor(gvar)}</div>

            {/* Expanded editor for selector options */}
            {editingId === gvar.id && (
              <div className="pt-2 border-t border-slate-800 space-y-3">
                <div>
                  <label className="block text-xs text-slate-500 mb-1">Description</label>
                  <input
                    type="text"
                    className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-sm text-white focus:outline-none focus:border-indigo-500"
                    value={gvar.description ?? ''}
                    onChange={(e) => updateGlobalVar(gvar.id, { description: e.target.value })}
                    placeholder="What this variable controls..."
                  />
                </div>
                {gvar.type === 'selector' && (
                  <div>
                    <label className="block text-xs text-slate-500 mb-1">Options (comma-separated)</label>
                    <input
                      type="text"
                      className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-sm text-white focus:outline-none focus:border-indigo-500"
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
